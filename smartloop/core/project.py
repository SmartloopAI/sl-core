import os
from smartloop.core.errors import ModelNotFound


class Project(object):
    """
       Creates a new project or use an existing one
    """

    def __init__(self, data_dir, project_id, model_id):
        self.data_dir = data_dir
        self.project_id = project_id
        self.model_id = model_id

    def get_project_dir(self, training=False):
        path = os.path.join(self.data_dir, self.project_id)

        if training:
            project_dir = self._get_project_dir()
            # create the directories
            if not os.path.isdir(project_dir):
                os.makedirs(project_dir)
            return project_dir
        else:
            project_dir = self._get_project_dir()
            # check if director exists
            if os.path.isdir(project_dir) and self._is_valid_model_folder(project_dir):
                return project_dir

            dirs = []
            for d in os.listdir(path):
                if os.path.isdir(os.path.join(path, d)) and d is not self.model_id:
                    # checking if the model is atomic
                    if self._is_valid_model_folder(os.path.join(path, d)):
                        dirs.append(d)

            if len(dirs) == 0:
                raise ModelNotFound("Invalid model: {}".format(self.model_id))

            return os.path.join(self.data_dir, self.project_id, sorted(
                dirs,
                key=lambda x: os.path.getctime("{}/{}".format('{}/{}'.format(self.data_dir, self.project_id), x)),
                reverse=True
            )[0])

    def _is_valid_model_folder(self, project_dir):
        files = ["model.h5"]
        return all([os.path.exists(os.path.join(project_dir, f)) for f in files])

    def _get_project_dir(self):
        return os.path.join(self.data_dir, self.project_id, self.model_id)
