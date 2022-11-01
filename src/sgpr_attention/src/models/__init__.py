from .SGPR_baseline import SGPR_Baseline
from .SGPR_geo_baseline import SGPR_Geo_Baseline
from .SGPR_geo_attention import SGPR_Geo_Attention
from .SGPR_attention import SGPR_Attention
from .SGPR_baseline_attention_fusion import SGPR_Baseline_Attention_Fusion
from .SGPR_attention_attention_fusion import SGPR_Attention_Attention_Fusion
from .SGPR_single import SGPR_Single
from .SGPR_attention_consistent import SGPR_Attention_Consistent
from .SGPR_geo_attention_attention_fusion import SGPR_Geo_Attention_Attention_Fusion
def get_model():
    return {"SGPR_Baseline":SGPR_Baseline,
            "SGPR_Geo_Baseline":SGPR_Geo_Baseline,
            "SGPR_Geo_Attention": SGPR_Geo_Attention,
            "SGPR_Attention":SGPR_Attention,
            "SGPR_Baseline_Attention_Fusion":SGPR_Baseline_Attention_Fusion,
            "SGPR_Attention_Attention_Fusion":SGPR_Attention_Attention_Fusion,
            "SGPR_Single":SGPR_Single,
            "SGPR_Attention_Consistent":SGPR_Attention_Consistent,
            "SGPR_Geo_Attention_Attention_Fusion":SGPR_Geo_Attention_Attention_Fusion
            }
