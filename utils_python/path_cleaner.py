import os

def convert_path_format(path: str, target_format: str) -> str:
    """Converts a directory path to a specified format (Linux or Windows)."""
    if target_format.lower() not in ['linux', 'windows']:
        raise ValueError("target_format must be 'linux' or 'windows'")

    if target_format.lower() == 'linux':
        path = path.replace('\\', '/')
        if not path.startswith('/media/'):
            path = '/media/' + path
        if "//" in path:
            path = path.replace('//', '/')
        if ":" in path:
            path = path.replace(':', '')
        return path

    elif target_format.lower() == 'windows':
        if path.startswith('/media/'):
            parts = path.split('/')
            if len(parts) > 2 and len(parts[2]) == 1:
                path = f"{parts[2]}:/" + "/".join(parts[3:])
        if "\\\\" in path:
            path = path.replace("\\\\", os.sep)
        return path.replace('/', os.sep)
