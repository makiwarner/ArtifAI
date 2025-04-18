def flatten_content(item):
    """
    Recursively traverse the item and return a list of all string values found.
    """
    results = []
    if isinstance(item, str):
        results.append(item.strip())
    elif isinstance(item, list):
        for element in item:
            results.extend(flatten_content(element))
    elif isinstance(item, dict):
        for key, value in item.items():
            results.extend(flatten_content(value))
    return results