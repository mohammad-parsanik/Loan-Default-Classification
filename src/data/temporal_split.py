import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def split_by_time(instances: List[Dict]) -> tuple:
    """
    Split the dataset chronologically based on snapshots.
    Assume 5 snapshots: 3 for train, 1 for val, 1 for test.
    """
    snapshots = sorted(list(set(inst['snapshot_date'] for inst in instances)))
    
    if len(snapshots) < 3:
        logger.warning(f"Only {len(snapshots)} snapshots found. Defaulting to random split or keeping all in train.")
        # Fallback to single split for testing
        return instances, [], []
        
    if len(snapshots) >= 5:
        train_snaps = snapshots[:-2]
        val_snaps = [snapshots[-2]]
        test_snaps = [snapshots[-1]]
    else:
        # e.g., 4 snapshots -> 2 train, 1 val, 1 test
        # e.g., 3 snapshots -> 1 train, 1 val, 1 test
        train_snaps = snapshots[:-2]
        val_snaps = [snapshots[-2]]
        test_snaps = [snapshots[-1]]
        
    logger.info(f"Temporal Split Configuration:")
    logger.info(f"  Train: {train_snaps}")
    logger.info(f"  Val:   {val_snaps}")
    logger.info(f"  Test:  {test_snaps}")
    
    train_inst = [i for i in instances if i['snapshot_date'] in train_snaps]
    val_inst = [i for i in instances if i['snapshot_date'] in val_snaps]
    test_inst = [i for i in instances if i['snapshot_date'] in test_snaps]
    
    logger.info(f"Instances -> Train: {len(train_inst)}, Val: {len(val_inst)}, Test: {len(test_inst)}")
    
    return train_inst, val_inst, test_inst
