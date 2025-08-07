import os
import os.path as osp
import logging
import datetime
import random
import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from preprocess import preprocess
from utils import (
    seed_torch, set_logger, Cfg, count_parameters, test_step, save_model
)
from dataset import LBSNDataset
from agents.agent import Agent

def create_batch_data(data_dict, batch_size):
    total_size = len(data_dict['label'])
    num_batches = (total_size + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)
        batch = {}
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[start_idx:end_idx]
            elif isinstance(v, list):
                batch[k] = v[start_idx:end_idx]
            else:
                batch[k] = v
        yield batch

def make_sampler(lbsn_dataset, split, limit_samples=None):
    node_idx = getattr(lbsn_dataset, f'node_idx_{split}')
    max_time = getattr(lbsn_dataset, f'max_time_{split}')
    label = getattr(lbsn_dataset, f'label_{split}')
    sample_idx = getattr(lbsn_dataset, f'sample_idx_{split}')
    weekday = getattr(lbsn_dataset, f'weekday_names_{split}')
    day_type = getattr(lbsn_dataset, f'day_type_{split}')
    poi_category = getattr(lbsn_dataset, f'poi_category_names_{split}')

    trajectory_keys = [pair['key'] for pair in lbsn_dataset.key_query_pairs[split]]
    trajectory_queries = [pair['query'] for pair in lbsn_dataset.key_query_pairs[split]]
    trajectory_ids = [pair['traj_id'] for pair in lbsn_dataset.key_query_pairs[split]]
    qa_pairs = lbsn_dataset.qa_pairs[split]

    if limit_samples is not None:
        if isinstance(node_idx, torch.Tensor):
            node_idx = node_idx[:limit_samples]
        if isinstance(max_time, torch.Tensor):
            max_time = max_time[:limit_samples]
        if isinstance(label, torch.Tensor):
            label = label[:limit_samples]
        if isinstance(sample_idx, torch.Tensor):
            sample_idx = sample_idx[:limit_samples]
        weekday = weekday[:limit_samples] if weekday else []
        day_type = day_type[:limit_samples] if day_type else []
        poi_category = poi_category[:limit_samples] if poi_category else []
        trajectory_keys = trajectory_keys[:limit_samples] if trajectory_keys else []
        trajectory_queries = trajectory_queries[:limit_samples] if trajectory_queries else []
        trajectory_ids = trajectory_ids[:limit_samples] if trajectory_ids else []
        qa_pairs = qa_pairs[:limit_samples] if qa_pairs else []

    return {
        'node_idx': node_idx,
        'max_time': max_time,
        'label': label,
        'sample_idx': sample_idx,
        'weekday': weekday,
        'day_type': day_type,
        'poi_category': poi_category,
        'trajectory_keys': trajectory_keys,
        'trajectory_queries': trajectory_queries,
        'trajectory_ids': trajectory_ids,
        'qa_pairs': qa_pairs,
    }

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    parser.add_argument('--multi_run_mode', help='Run multiple experiments with the same config.', action='store_true')
    parser.add_argument('--debug', help='Enable debug mode with limited samples for train/valid/test', action='store_true')
    parser.add_argument('--debug_samples', help='Number of samples in debug mode', type=int, default=20)
    args = parser.parse_args()
    cfg = Cfg(args.yaml_file)

    device = f'cuda:{cfg.run_args.gpu}' if cfg.run_args.gpu >= 0 else 'cpu'
    cfg.run_args.device = device

    if args.multi_run_mode:
        cfg.run_args.seed = None
    if cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)
    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.run_args.save_path = f'tensorboard/{current_time}/{cfg.dataset_args.dataset_name}'
    cfg.run_args.log_path = f'log/{current_time}/{cfg.dataset_args.dataset_name}'
    os.makedirs(cfg.run_args.save_path, exist_ok=True)
    os.makedirs(cfg.run_args.log_path, exist_ok=True)
    set_logger(cfg.run_args)
    summary_writer = SummaryWriter(log_dir=cfg.run_args.save_path)

    preprocess(cfg)
    lbsn_dataset = LBSNDataset(cfg)
    df = lbsn_dataset.get_df()
    train_df = lbsn_dataset.get_train_df()
    valid_df = lbsn_dataset.get_valid_df()
    test_df = lbsn_dataset.get_test_df()

    train_limit = args.debug_samples if args.debug else None
    valid_limit = args.debug_samples if args.debug else None
    test_limit = args.debug_samples if args.debug else None

    agent = Agent(
        cfg,
        df=df,
        df_train=train_df,
        df_valid=valid_df,
        df_test=test_df,
        dataset_instance=lbsn_dataset
    ).to(device)

    logging.info(f'[Training] Seed: {seed}')
    logging.info(f'[Training] #Parameters: {count_parameters(agent)}')

    if cfg.run_args.do_train:
        sampler_train = make_sampler(lbsn_dataset, 'train', limit_samples=train_limit)
        global_step = 0
        best_metrics = 0.0

        for epoch in range(cfg.run_args.epoch):
            if global_step >= cfg.run_args.max_steps:
                break

            agent.train()
            batch_iterator = create_batch_data(sampler_train, cfg.run_args.batch_size)
            for batch_idx, raw_batch in enumerate(batch_iterator):
                batch = {}
                for k, v in raw_batch.items():
                    if isinstance(v, (torch.Tensor, torch.nn.Module)):
                        batch[k] = v.to(device)
                    else:
                        batch[k] = v

                recommendations, category_recommendations = agent.run(
                    batch,
                    strategy=cfg.model_args.reflection_strategy,
                    split='train'
                )

                logging.info(f"[Epoch {epoch}] Step {global_step}: Recommendations: {recommendations}")

                global_step += 1
                if global_step >= cfg.run_args.max_steps:
                    break

            if cfg.run_args.do_validate and global_step % cfg.run_args.valid_steps == 0:
                sampler_validate = make_sampler(lbsn_dataset, 'valid', limit_samples=valid_limit)
                logging.info(f'[Evaluating] Epoch {epoch}, step {global_step}:')
                recall_res, ndcg_res, map_res, mrr_res, category_recalls, category_ndcgs, category_maps, category_mrr = test_step(
                    agent,
                    data=sampler_validate,
                    split='valid'
                )

                logging.info(f"[Valid] Recall@1: {recall_res[1]}, Recall@20: {recall_res[20]}")
                summary_writer.add_scalar(f'validate/Recall@1', 100 * recall_res[1], global_step)
                summary_writer.add_scalar(f'validate/Recall@5', 100 * recall_res[5], global_step)
                summary_writer.add_scalar(f'validate/Recall@10', 100 * recall_res[10], global_step)
                summary_writer.add_scalar(f'validate/Recall@20', 100 * recall_res[20], global_step)
                summary_writer.add_scalar(f'validate/MRR', mrr_res, global_step)
                summary_writer.add_scalar(f'validate/Category_Recall@1', 100 * category_recalls[1], global_step)
                summary_writer.add_scalar(f'validate/Category_Recall@20', 100 * category_recalls[20], global_step)
                summary_writer.add_scalar(f'validate/Category_MRR', category_mrr, global_step)

                metrics = 4 * recall_res[1] + recall_res[20]
                if metrics > best_metrics:
                    save_variable_list = {
                        'step': global_step,
                        'high_level_memory': agent.high_level_memory,
                        'low_level_memory': agent.low_level_memory,
                        'reflections': getattr(agent, 'reflections', None),
                    }
                    logging.info(f'[Training] Save model at step {global_step} epoch {epoch}')
                    best_metrics = metrics

    if cfg.run_args.do_test:
        sampler_test = make_sampler(lbsn_dataset, 'test', limit_samples=test_limit)
        logging.info('[Evaluating] Start evaluating on test set...')
        recall_res, ndcg_res, map_res, mrr_res, category_recalls, category_ndcgs, category_maps, category_mrr = test_step(
            agent,
            data=sampler_test,
            split='test'
        )
        num_params = count_parameters(agent)
        metric_dict = {
            'hparam/num_params': num_params,
            'hparam/Recall@1': recall_res[1],
            'hparam/Recall@5': recall_res[5],
            'hparam/Recall@10': recall_res[10],
            'hparam/Recall@20': recall_res[20],
            'hparam/NDCG@1': ndcg_res[1],
            'hparam/NDCG@5': ndcg_res[5],
            'hparam/NDCG@10': ndcg_res[10],
            'hparam/NDCG@20': ndcg_res[20],
            'hparam/MAP@1': map_res[1],
            'hparam/MAP@5': map_res[5],
            'hparam/MAP@10': map_res[10],
            'hparam/MAP@20': map_res[20],
            'hparam/MRR': mrr_res,
            'hparam/Category_Recall@1': category_recalls[1],
            'hparam/Category_Recall@20': category_recalls[20],
            'hparam/Category_NDCG@20': category_ndcgs[20],
            'hparam/Category_MRR': category_mrr,
        }
        logging.info(f'[Evaluating] Test evaluation result: {metric_dict}')
        summary_writer.close()

if __name__ == '__main__':
    main()