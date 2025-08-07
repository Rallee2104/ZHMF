import torch
import time
import logging
from torch import nn

from .prompting import build_high_level_prompt, build_low_level_prompt
from .memory import reflect
from .memory import retrieve_similar_reflections
from .feature_extract import update_trajectory_history, extract_relevant_trajectories
from .agent_utils import parse_single_recommendation, is_correct, extract_category_prediction
from .caching import output_prediction, load_prediction
from .llm import LlamaLLM

class Agent(nn.Module):
    def __init__(self, config, df=None, df_train=None, df_valid=None, df_test=None, dataset_instance=None):
        super().__init__()
        self.config = config
        self.device = config.model_args.device
        self.step_n = 0
        self.max_steps = config.model_args.max_steps
        self.finished = False
        self.reflection_frequency = config.model_args.reflection_frequency

        self.scratchpad = ""
        self.reflections = []
        self.reflection_embeddings = []
        self.trajectory_history = []
        self.reflections_str = ''
        self.last_attempt = ''

        self.high_level_memory = []
        self.low_level_memory = []
        self.high_reflection_embeddings = []
        self.low_reflection_embeddings = []

        model_args = config.model_args
        self.llm = LlamaLLM(
            model_args=model_args,
            api_key=None,
            base_url="http://127.0.0.1:8002",
            model="Meta-Llama-3-8B-Instruct",
        )

        self.max_reflection_tokens = config.model_args.max_reflection_tokens

        self.df = df
        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.dataset = dataset_instance

        self.dataset_dict = {
            'train': self.df_train,
            'valid': self.df_valid,
            'test': self.df_test
        }
        self.current_split = None

        from sentence_transformers import SentenceTransformer
        local_embedding_path = "./downloaded_embedding"
        self.vector_model = SentenceTransformer(local_embedding_path)

        self.user_trajectories = []
        self.user_trajectory_embeddings = []

        if dataset_instance:
            for split in ['train', 'valid', 'test']:
                qa_pairs = dataset_instance.qa_pairs.get(split, [])
                for qa_pair in qa_pairs:
                    question = qa_pair.get('question', '').replace('<question>: ', '')
                    answer = qa_pair.get('answer', '').replace('<answer>: ', '')
                    full_trajectory = f"{question} {answer}"
                    self.user_trajectories.append(full_trajectory)
            if self.user_trajectories:
                self.user_trajectory_embeddings = self.vector_model.encode(self.user_trajectories)

    def run(self, check_in_data, strategy="REFLEXION", split='train'):
        self.current_split = split
        self.reset()
        batch_size = check_in_data['label'].shape[0]
        device = self.device

        recommendations = torch.zeros((batch_size, self.config.model_args.top_k), dtype=torch.long, device=device)
        category_recommendations = []

        qa_pairs = check_in_data['qa_pairs']
        if not qa_pairs:
            return recommendations, category_recommendations

        for i in range(min(batch_size, len(qa_pairs))):
            logging.info(f"[agent:run] Processing sample {i+1}")
            qa_pair = qa_pairs[i]
            trajectory = qa_pair['metadata']['trajectory_id']

            # Using cached prediction
            poi_prediction, category_prediction = load_prediction(self, trajectory)
            if poi_prediction is not None and category_prediction is not None:
                logging.info(f"[agent:run] Using cached prediction result: {trajectory}")
                target_poi = qa_pair['metadata']['target_poi']
                is_right = is_correct(self, poi_prediction, target_poi)
                logging.info(f"[agent:run][Cache Check] Predicted POI is correct: {is_right}")
                recommendations[i, :len(poi_prediction)] = torch.tensor(poi_prediction, dtype=torch.long).to(self.device)
                category_recommendations.append(category_prediction if category_prediction else [])
                continue

            # Trajectory history management
            update_trajectory_history(self, qa_pair)
            self.scratchpad = f"""Question: {qa_pair['question']}
            Answer: {qa_pair.get('answer', 'No answer available')}"""

            relevant_trajectories = extract_relevant_trajectories(self, qa_pair['question'])

            high_level_prompt = build_high_level_prompt(self, qa_pair, relevant_trajectories)
            token_count = self.llm.count_tokens(high_level_prompt)
            if token_count > self.config.model_args.max_input_length:
                high_level_prompt = self.llm.truncate_text(high_level_prompt, self.config.model_args.max_input_length)

            raw_prediction = self.llm(high_level_prompt, max_output_length=128).strip()

            category_prediction = extract_category_prediction(raw_prediction, self.dataset)
            category_recommendations.append(category_prediction)
            true_category = qa_pair['metadata']['target_category']
            logging.info(f"[agent:run][High Level Prediction] Predicted categories: {category_prediction}")
            logging.info(f"[agent:run][High Level Prediction] True category: {true_category}")

            prediction_correct = true_category in category_prediction
            logging.info(f"[agent:run][High Level Prediction] Prediction is correct: {prediction_correct}")

            if strategy != "NONE":
                logging.info(f"[agent:run][High Level Reflection] {'Success' if prediction_correct else 'Failure'} reflection")
                reflect(
                    self,
                    qa_pair,
                    category_prediction,
                    true_category,
                    strategy,
                    reflection_level="high",
                    raw_prediction=raw_prediction,
                    is_success=prediction_correct
                )

            low_level_prompt = build_low_level_prompt(self, qa_pair, category_prediction, relevant_trajectories)
            token_count = self.llm.count_tokens(low_level_prompt)
            if token_count > self.config.model_args.max_input_length:
                low_level_prompt = self.llm.truncate_text(low_level_prompt, self.config.model_args.max_input_length)

            poi_prediction_text = self.llm(low_level_prompt, max_output_length=256).strip()

            poi_ids = parse_single_recommendation(self, poi_prediction_text)
            logging.info(f"[agent:run][Low Level Prediction] Predicted POI IDs: {poi_ids}")

            for k, poi_id in enumerate(poi_ids):
                if poi_id >= 0:
                    recommendations[i, k] = poi_id

            output_prediction(self, recommendations[i], trajectory, category_prediction)

            target_poi = qa_pair['metadata']['target_poi']
            if strategy != "NONE":
                is_right = is_correct(self, poi_ids, target_poi)
                logging.info(f"[agent:run][Low Level Reflection] {'Success' if is_right else 'Failure'} reflection")
                reflect(
                    self,
                    qa_pair,
                    poi_ids,
                    target_poi,
                    strategy,
                    reflection_level="low",
                    predicted_categories=category_prediction,
                    is_success=is_right
                )

        return recommendations, category_recommendations

    def reset(self):
        self.step_n = 0
        self.finished = False
        self.scratchpad = ""

    def add_trajectory(self, trajectory):
        self.user_trajectories.append(trajectory)

    def _format_reflections(self, current_trajectory, reflection_level):
        top_reflections_num = 2
        similar_reflections = retrieve_similar_reflections(
            self,
            current_trajectory,
            reflection_level
        )
        formatted_reflections = [
            f"{i}. {reflection}"
            for i, reflection in enumerate(similar_reflections[:top_reflections_num], 1)
        ]
        return "\n".join(formatted_reflections)