from .semantickitti import SemanticKitti
from .semantickitti_globalcenter import SemanticKittiGlobalCenter
from .semantickitti_bbox import SemanticKittiBbox
from .semantickitti_bbox_ros import SemanticKittiBboxROS
def get_dataset():
    return {
        "SemanticKitti":SemanticKitti,
        "SemanticKittiGlobalCenter":SemanticKittiGlobalCenter,
        "SemanticKittiBbox":SemanticKittiBbox,
        "SemanticKittiBboxROS":SemanticKittiBboxROS
            }