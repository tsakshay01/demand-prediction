import sys
import traceback
try:
    from model import MultimodalDemandModel
    print('Init...')
    m = MultimodalDemandModel()
    print('Success')
except:
    traceback.print_exc()
