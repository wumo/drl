import json

def toJson(obj, max_depth=None, depth=0):
    if max_depth is not None and depth >= max_depth:
        return str(obj)
    depth += 1
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {toJson(k, max_depth, depth): toJson(v, max_depth, depth)
                    for k, v in obj.items()}
        
        elif isinstance(obj, tuple):
            return (toJson(x, max_depth, depth) for x in obj)
        
        elif isinstance(obj, list):
            return [toJson(x, max_depth, depth) for x in obj]
        
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return toJson(obj.__name__, max_depth, depth)
        
        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {toJson(k, max_depth, depth): toJson(v, max_depth, depth)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}
        
        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
