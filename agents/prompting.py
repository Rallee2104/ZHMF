import logging
from dataset.behavior_analysis import get_trajectory_history

def validate_inputs(agent, qa_pair, log_prefix=""):
    """Validate common inputs for prompt building functions"""
    metadata = qa_pair.get('metadata', {})
    user_id = metadata.get('user_id')
    predict_time = metadata.get('timestamp')
    now_poi = metadata.get('now_poi')
    
    if user_id is None or predict_time is None or now_poi is None:
        logging.error(f"{log_prefix}QA pair metadata missing required information")
        return None, None, None, None
        
    current_df = agent.df
    if current_df is None:
        logging.error(f"{log_prefix}Invalid dataset")
        return None, None, None, None
        
    if agent.dataset is None:
        logging.error(f"{log_prefix}Dataset instance not available")
        return None, None, None, None
        
    return user_id, predict_time, now_poi, current_df

def build_high_level_prompt(agent, qa_pair, relevant_trajectories, num_categories=20):
    """
    Build high-level strategy prompt for category prediction.
    """
    validation = validate_inputs(agent, qa_pair, "[High-Level Prompt] ")
    if validation[0] is None:
        return ""
        
    user_id, predict_time, now_poi, current_df = validation
    metadata = qa_pair.get('metadata', {})
    
    try:
        user_patterns = agent.dataset.extract_user_patterns(
            user_id=user_id,
            df=current_df,
            now_poi=now_poi,
            predict_time=predict_time,
            window_size=10,
            metadata=metadata,
            layer="high"
        )

        top_categories = user_patterns['long_term_preference']['top_categories']
        current_categories = user_patterns['current_preference']['top_categories']

        category_patterns = []
        for category, stats in top_categories:
            pattern = (
                f"{category} ({stats['total_visits']} total visits - "
                f"Weekday: {stats['weekday_visits']}, "
                f"Weekend: {stats['weekend_visits']}, "
                f"Peak hour: {stats['peak_hour']}:00)"
            )
            category_patterns.append(pattern)
        current_categories_str = ", ".join([
            f"{cat} ({count} visits)"
            for cat, count in current_categories
        ])
        sequence_transition = user_patterns.get('sequence_transition', {})
        transitions = sequence_transition.get('top_transitions', [])
        transition_patterns = []
        if transitions:
            transition_patterns = [
                f"- {t['pattern']} (probability: {t['probability']:.2f}, count: {t['count']})"
                for t in transitions
            ]
        poi_distances = user_patterns.get('poi_distances', [])
        if poi_distances:
            candidate_set = [
                f"({d['poi_id']}, {d['distance']:.4f}, {d['category']})"
                for d in poi_distances
            ]
            candidate_set_str = ', '.join(candidate_set)
        else:
            candidate_set_str = ""

        prompt = f"""As an intelligent POI recommendation system, analyze the following user behavior data to predict the next {num_categories} most likely POI categories.

Context:
Time: {predict_time}
Question: {qa_pair['question'].replace('<question>: ', '')}

Historical Behavior Analysis:
• Most Active Categories and Patterns:
{chr(10).join(f"  - {p}" for p in category_patterns)}

• Short-Term Top Categories: {current_categories_str}

• Common Transition Patterns:
{chr(10).join(f"  {p}" for p in transition_patterns)}

Similar User Trajectories:
{', '.join(relevant_trajectories)}

Candidate POIs [ID, Distance(km), Category]:
{candidate_set_str}

Task:
1. Analyze the user's historical preferences and current context
2. Consider temporal patterns and geographical constraints
3. Evaluate category transition probabilities
4. Output exactly {num_categories} category names, ranked by likelihood, without any explanation"""

        if agent.high_level_memory:
            formatted_memory = agent._format_reflections(
                qa_pair['question'],
                reflection_level="high"
            )
            prompt = f"{prompt}\n\nRelevant Historical Reflections:\n{formatted_memory}"
        if num_categories > 2:
            format_examples = f"1. [Category Name]\n2. [Category Name]\n...\n{num_categories}. [Category Name]"
        else:
            format_examples = "\n".join([f"{i+1}. [Category Name]" for i in range(num_categories)])
        prompt = f"""{prompt}

Format your response as:
{format_examples}"""
        return prompt
    except Exception as e:
        logging.error(f"Error building high-level prompt: {str(e)}")
        return ""

def build_low_level_prompt(agent, qa_pair, predicted_category, relevant_trajectories):
    """
    Build low-level strategy prompt for specific POI recommendations.
    """
    validation = validate_inputs(agent, qa_pair, "[Low-Level Prompt] ")
    if validation[0] is None:
        return ""
        
    user_id, predict_time, now_poi, current_df = validation
    metadata = qa_pair.get('metadata', {})

    user_patterns = agent.dataset.extract_user_patterns(
        user_id=user_id,
        df=current_df,
        now_poi=now_poi,
        predict_time=predict_time,
        window_size=10,
        metadata=metadata,
        layer="low",
        predicted_categories=predicted_category
    )

    user_profile = user_patterns.get('user_profile', {})
    geo_pattern = user_patterns.get('geospatial_pattern', {})
    current_prefs = user_patterns.get('current_preference', {})
    recent_visits = current_prefs.get('recent_visits', [])[:5]

    trajectory_history = get_trajectory_history(
        user_id=user_id,
        df=current_df,
        predict_time=predict_time
    )

    question = qa_pair['question'].replace('<question>: ', '')

    poi_distances = user_patterns.get('poi_distances', [])
    if poi_distances:
        candidate_set = [f"({d['poi_id']}, {d['distance']:.4f}, {d['category']})" for d in poi_distances]
        candidate_set_str = ', '.join(candidate_set)
    else:
        candidate_set_str = ""

    prompt = f"""As an intelligent POI recommendation system, generate specific POI recommendations based on the following data analysis.

Current Scenario:
• Target Categories: {', '.join(predicted_category)}
• Time: {predict_time}
• Question: {question}

User Profile:
Recent Activities:
{', '.join(recent_visits)}

Behavioral Patterns:
• Day & Time Preferences [ID, Category, Distance(km), visits, peak time, day type]: {user_patterns['behavioral_patterns']['day_preferences']}
• Area Preferences: {user_patterns['behavioral_patterns']['area_preferences']}
• Visit Radius: {user_patterns['behavioral_patterns']['visit_radius']}

Historical Movement:
• Past Trajectories: {trajectory_history}
• Similar User Paths: {', '.join(relevant_trajectories)}

Candidate POIs [ID, Distance(km), Category]:
{candidate_set_str}

Task:
1. Focus on POIs within {predicted_category} category
2. Consider distance and accessibility
3. Account for time-based preferences
4. Balance between popular and personalized choices
5. Generate 20 different ranked recommendations without any explanation"""

    if agent.low_level_memory:
        current_trajectory = qa_pair['question']
        formatted_memory = agent._format_reflections(
            current_trajectory,
            reflection_level="low"
        )
        prompt = f"{prompt}\n\nRelevant Historical Reflections:\n{formatted_memory}"

    prompt = f"""{prompt}

Format your response as: 
RANK 1: [POI-ID, Category] 
RANK 2: [POI-ID, Category] 
... 
RANK 20: [POI-ID, Category]"""

    return prompt