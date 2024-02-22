import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pdb

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["custom_scalars"]
        pdb.set_trace()
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
    return runlog_data

path="../runs/nsol25/hv_3obj_ref111_flow_init_sigma1_15112023_144512/fold0/run0/events.out.tfevents.1700055914.galaxy01.scilens.private.1054088.0" #folderpath
df=tflog2pandas(path)
#df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
df.to_csv("../outputs/run_summary.csv")