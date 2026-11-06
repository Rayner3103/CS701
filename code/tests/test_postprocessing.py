import importlib
import importlib.util
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_native_postprocessing_module_exists():
    assert importlib.util.find_spec('native_postprocessing') is not None


def test_fusion_and_empty_proposals():
    module=importlib.import_module('native_postprocessing')
    logits=torch.tensor([1.,0.,-5.,-5.,-5.,-5.,-5.,-5.,-5.])
    assert module.fused_class(logits,{'raw_logits':torch.empty(0,10)})==0
    raw=torch.full((2,10),-20.);raw[:,2]=20
    assert module.fused_class(logits,{'raw_logits':raw})==1
    assert module.FINAL_POST['shift']==0 and module.FINAL_POST['total']==100
