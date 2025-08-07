import time
import logging
from typing import List
from dataset.behavior_analysis import get_trajectory_history

def add_reflection_to_index(agent, reflection_data, reflection_level):
    reflection_text = reflection_data['reflection']
    embedding = agent.vector_model.encode([reflection_text])[0]
    if reflection_level == "high":
        agent.high_level_memory.append(reflection_text)
        agent.high_reflection_embeddings.append(embedding)
    elif reflection_level == "low":
        agent.low_level_memory.append(reflection_text)
        agent.low_reflection_embeddings.append(embedding)
    logging.info(f"Current storage status - High level: {len(agent.high_level_memory)} entries({len(agent.high_reflection_embeddings)} embeddings), "
                 f"Low level: {len(agent.low_level_memory)} entries({len(agent.low_reflection_embeddings)} embeddings)")

def retrieve_similar_reflections(agent, current_trajectory, reflection_level):
    if reflection_level == "high":
        reflection_embeddings = agent.high_reflection_embeddings
        reflections = agent.high_level_memory
    else:
        reflection_embeddings = agent.low_reflection_embeddings
        reflections = agent.low_level_memory

    if not reflection_embeddings:
        return []

    current_embedding = agent.vector_model.encode([current_trajectory])[0]
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    reflection_embeddings_np = np.vstack(reflection_embeddings)
    similarities = cosine_similarity(
        current_embedding.reshape(1, -1),
        reflection_embeddings_np
    )
    similar_indices = np.argsort(similarities[0])[::-1]
    top_k_indices = similar_indices[:2]
    similar_reflections = [reflections[i] for i in top_k_indices]
    return similar_reflections

def reflect(agent, qa_pair, prediction, actual,
            strategy, reflection_level,
            raw_prediction=None,
            predicted_categories=None, is_success=False):
    """
    Generate reflection text and store it in the specified memory, supporting high and low levels.
    """
    if strategy == "NONE":
        return

    if reflection_level == "high":
        metadata = qa_pair.get('metadata', {})
        user_id = metadata.get('user_id')
        predict_time = metadata.get('timestamp')
        now_poi = metadata.get('now_poi')

        user_patterns = agent.dataset.extract_user_patterns(
            user_id=user_id,
            df=agent.df,
            now_poi=now_poi,
            predict_time=predict_time,
            window_size=10,
            metadata=metadata,
            layer="high"
        )
        long_term_categories = user_patterns['long_term_preference'].get('top_categories', [])
        current_categories = user_patterns['current_preference'].get('top_categories', [])

        long_term_patterns = user_patterns.get('long_term_preference', {})
        category_stats = long_term_patterns.get('category_stats', {})
        visit_pattern = long_term_patterns.get('visit_pattern', {})

        most_frequent_category = max(
            category_stats.items(),
            key=lambda x: x[1]['total_visits']
        )[0] if category_stats else "Unknown"
        most_frequent_category_count = category_stats[most_frequent_category]['total_visits'] if most_frequent_category != "Unknown" else 0

        peak_hours = visit_pattern.get('peak_hours', {})
        peak_hour = max(peak_hours, key=peak_hours.get) if peak_hours else "Unknown"
        peak_hour_value = peak_hours.get(peak_hour, 0)

        current_patterns = user_patterns.get('current_preference', {})
        recent_categories = current_patterns.get('recent_categories', {})
        most_recent_category = max(recent_categories, key=recent_categories.get) if recent_categories else "Unknown"
        most_recent_category_count = recent_categories.get(most_recent_category, 0)

        sequence_transition = user_patterns.get('sequence_transition', {})
        transition_probs = sequence_transition.get('transition_probs', {})

        if transition_probs:
            most_common_transition = max(transition_probs, key=transition_probs.get)
            most_common_transition_prob = transition_probs[most_common_transition]
            sequence_info = f"Most common category transition: {most_common_transition} (probability: {most_common_transition_prob:.2f})"
        else:
            sequence_info = "Most common category transition: No sequence data available"

        task_description = """
Task: Provide a single-sentence reflection on the category-level recommendation that:
1. Identifies specific factors (e.g., historical preferences, peak times, transitions) that led to successful category prediction.
2. Explains how user's historical preferences and recent patterns (e.g., time, location, transitions) contributed to accuracy.
3. Highlights how spatial-temporal patterns and category transitions were effectively leveraged.
4. Summarizes actionable best practices for improving future similar predictions.
""" if is_success else """
Task: Provide a single-sentence reflection on the category-level recommendation that:
1. Analyzes the gap between predicted and actual categories.
2. Considers user's historical preferences and recent behavior changes.
3. Evaluates the impact of time patterns and category transitions.
4. Suggests one key improvement for future recommendations.
"""

        high_level_prompt = f"""
Time: {predict_time}
Question: {qa_pair['question']}

Historical Behavior Analysis:
• Most Active Categories and Patterns:
{chr(10).join(f"  - {cat} ({stats['total_visits']} total visits - Weekday: {stats['weekday_visits']}, Weekend: {stats['weekend_visits']}, Peak hour: {stats['peak_hour']}:00)" for cat, stats in long_term_categories)}
• Short-Term Top Categories: {', '.join([f"{cat} ({count} visits)" for cat, count in current_categories])}
• Common Transition Patterns:
{chr(10).join(f"  - {t['pattern']} (probability: {t['probability']:.2f}, count: {t['count']})" for t in sequence_transition.get('top_transitions', []))}

Prediction Analysis:
- Predicted Categories: {prediction}
- Actual Category: {actual}
- Prediction Status: {'Successful' if is_success else 'Failed'}
{task_description}
Keep your reflection focused and under 50 words.
"""

        high_level_reflection = agent.llm(high_level_prompt, max_output_length=64).strip()
        logging.info(f"[agent:run][Reflection Result] Generated high-level reflection:\n{high_level_reflection}")
        agent.high_level_memory.append(high_level_reflection)

        add_reflection_to_index(
            agent,
            {'reflection': high_level_reflection},
            reflection_level="high"
        )

    elif reflection_level == "low":
        metadata = qa_pair.get('metadata', {})
        user_id = metadata.get('user_id')
        predict_time = metadata.get('timestamp')
        now_poi = metadata.get('now_poi')

        user_patterns = agent.dataset.extract_user_patterns(
            user_id=user_id,
            df=agent.df,
            now_poi=now_poi,
            predict_time=predict_time,
            window_size=10,
            metadata=metadata,
            layer="low",
            predicted_categories=predicted_categories
        )
        current_prefs = user_patterns.get('current_preference', {})
        recent_visits = current_prefs.get('recent_visits', [])[:5]

        user_profile = user_patterns.get('user_profile', {})
        day_type_preferences = user_profile.get('day_type_preferences', {})

        geo_pattern = user_patterns.get('geospatial_pattern', {})
        top_categories = geo_pattern.get('top_categories', {})
        visit_radius = geo_pattern.get('visit_radius', "Unknown")

        trajectory_history = get_trajectory_history(
            user_id=user_id,
            df=agent.df,
            predict_time=predict_time
        )

        task_description = """
Task: Provide a single-sentence reflection on the POI-level recommendation that:
1. Identifies specific factors (e.g., recent visits, spatial-temporal patterns, visit radius) that led to successful POI prediction.
2. Explains how user's historical preferences and behavioral patterns (e.g., time, category transitions) contributed to accuracy.
3. Highlights how spatial-temporal alignment and visit radius were effectively leveraged.
4. Summarizes actionable best practices for future similar predictions.
""" if is_success else """
Task: Provide a single-sentence reflection on the POI-level recommendation that:
1. Evaluates the spatial-temporal accuracy of predicted POIs.
2. Considers distance from current location and typical visit radius.
3. Analyzes alignment with user's recent visit patterns.
4. Identifies one key factor for improving POI selection.
"""

        low_level_prompt = f"""
Current Scenario:
• Time: {predict_time}
• Question: {qa_pair['question']}
• Current Location: {now_poi}

User Profile:
Recent Activities:
{', '.join(recent_visits)}

Behavioral Patterns:
• Day & Time Preferences [ID, Category, Distance(km), visits, peak time, day type]: {user_patterns['behavioral_patterns']['day_preferences']}
• Area Preferences: {user_patterns['behavioral_patterns']['area_preferences']}
• Visit Radius: {user_patterns['behavioral_patterns']['visit_radius']}

Historical Movement:
• Past Trajectories: {trajectory_history}

Prediction Analysis:
- Predicted POIs: {prediction}
- Actual POI: {actual}
- Prediction Status: {'Successful' if is_success else 'Failed'}
- Distance: {next((poi['distance'] for poi in user_patterns.get('poi_distances', []) if poi['poi_id'] == actual), 'Unknown')}
{task_description}
Keep your reflection focused and under 50 words.
"""

        low_level_reflection = agent.llm(low_level_prompt, max_output_length=64).strip()
        logging.info(f"[agent:run][Reflection Result] Generated low-level reflection:\n{low_level_reflection}")
        agent.low_level_memory.append(low_level_reflection)

        add_reflection_to_index(
            agent,
            {'reflection': low_level_reflection},
            reflection_level="low"
        )