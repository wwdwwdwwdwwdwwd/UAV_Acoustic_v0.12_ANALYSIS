from __future__ import annotations

def build_semantic_detector(fs:int,config:dict):
    adapter_id=config["open_source_detector_adapter"]["adapter_id"]
    if adapter_id=="os01_naccache_crnn":
        from .naccache_crnn import NaccacheCRNNAdapter; return NaccacheCRNNAdapter(fs,config)
    if adapter_id=="os02_samid_ast":
        from .samid_ast import SamidASTAdapter; return SamidASTAdapter(fs,config)
    if adapter_id=="os03_echohawk_cnn":
        from .echohawk_cnn import EchoHawkCNNAdapter; return EchoHawkCNNAdapter(fs,config)
    if adapter_id=="os04_joules_feature_fusion":
        from .joules_feature_fusion import JoulesFeatureFusionAdapter; return JoulesFeatureFusionAdapter(fs,config)
    raise ValueError(f"unsupported semantic adapter: {adapter_id}")
