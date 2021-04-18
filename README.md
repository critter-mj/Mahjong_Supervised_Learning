## Train model
```
    python train.py --action_type TYPE
```
Here, TYPE is one of dahai, reach, chi, pon, kan.
Note that numpy feature must be dumped using akochan_ui, and variables test_prefix and train_prefix are set appropriately.
A file train_tmp.pth will be created and updated in several steps of learning.

## Dump cpu model
```
    python train.py --purpose dump_cpu_model --action_type TYPE
```
Dump cpu state_dict of supervised model from train_tmp.pth, which is applicable to Supervised_AI of akochan_ui.