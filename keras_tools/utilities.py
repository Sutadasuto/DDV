import os


def get_modality_views(modality_folder):

    views = sorted([os.path.join(modality_folder, f) for f in os.listdir(modality_folder)
                      if os.path.isdir(os.path.join(modality_folder, f)) and not f.startswith('.')
                    and f.lower() != "all"],
                     key=lambda f: f.lower())
    return tuple(views)
