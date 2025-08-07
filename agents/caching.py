import os
import json
import logging

def output_prediction(agent, prediction, trajectory, category_prediction=None):
    """
    Save prediction results
    """
    path = f'./save_output_nyc/{agent.current_split}/{trajectory}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(prediction, "tolist"):
        pred_list = prediction.tolist()
    else:
        pred_list = list(prediction)
    save_data = {
        "poi_prediction": pred_list,
        "category_prediction": category_prediction
    }
    try:
        with open(path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(save_data, ensure_ascii=False))
    except Exception as e:
        logging.error(f"[output_prediction] Save failed: {path}, reason: {e}")

def load_prediction(agent, trajectory):
    """
    Load prediction results for a specific trajectory
    """
    path = f'./save_output_nyc/{agent.current_split}/{trajectory}'
    if os.path.isfile(path) and os.access(path, os.R_OK):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                prediction_data = json.loads(file.read())
                poi_prediction = prediction_data.get("poi_prediction", [])
                category_prediction = prediction_data.get("category_prediction", [])
                if not poi_prediction:
                    logging.warning(f"[load_prediction] File exists but POI prediction is empty: {path}")
                    return None, None
                return poi_prediction, category_prediction
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"[load_prediction] File read failed: {path}, reason: {e}")
            return None, None
    else:
        return None, None