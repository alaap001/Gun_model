from fastai.vision import *   

learn = load_learner('path to folder containing pkl file','name_of.pkl', test=ImageList.from_folder('path to test folder'),bs=8)
preds,y = learn.get_preds(ds_type=DatasetType.Test)
