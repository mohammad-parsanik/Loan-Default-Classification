import logging
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

def split_by_time(instances: List[Dict]) -> tuple:
    """
    Split the dataset chronologically based on snapshots.
    Filters out snapshots that are less than 6 months old, as their labels
    are not fully mature (assuming a 6-month default horizon).
    Assume 5 snapshots: 3 for train, 1 for val, 1 for test.
    """
    all_snapshots = sorted(list(set(inst['snapshot_date'] for inst in instances)))
    
    # Filter snapshots to only include those at least 6 months old
    snapshots = []
    current_date = datetime.now()
    month = current_date.month - 6
    year = current_date.year
    if month <= 0:
        month += 12
        year -= 1
    
    day = current_date.day
    while True:
        try:
            six_months_ago = datetime(year, month, day)
            break
        except ValueError:
            day -= 1
            
    for snap in all_snapshots:
        snap_date = datetime.strptime(str(snap), "%Y%m%d")
        if snap_date <= six_months_ago:
            snapshots.append(snap)
        else:
            logger.warning(f"Excluding snapshot {snap} because it is newer than 6 months ago ({six_months_ago.strftime('%Y%m%d')}).")
    
    if len(snapshots) < 3:
        logger.warning(f"Only {len(snapshots)} valid snapshots found (out of {len(all_snapshots)} total). Defaulting to keeping all valid in train.")
        valid_inst = [i for i in instances if i['snapshot_date'] in snapshots]
        return valid_inst, [], []
        
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
        
    logger.info(f"Temporal Split Configuration (excluded {len(all_snapshots) - len(snapshots)} recent snapshots):")
    logger.info(f"  Train: {train_snaps}")
    logger.info(f"  Val:   {val_snaps}")
    logger.info(f"  Test:  {test_snaps}")
    
    train_inst = [i for i in instances if i['snapshot_date'] in train_snaps]
    val_inst = [i for i in instances if i['snapshot_date'] in val_snaps]
    test_inst = [i for i in instances if i['snapshot_date'] in test_snaps]
    
    logger.info(f"Instances -> Train: {len(train_inst)}, Val: {len(val_inst)}, Test: {len(test_inst)}")
    
    return train_inst, val_inst, test_inst
