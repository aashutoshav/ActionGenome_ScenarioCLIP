from dataloaders.vidor_dataset import VidOrDataset
from dataloaders.epic_kitchen_dataset import EpicKitchenDataset

def main():
    root_dir = "/scratch/saali/Datasets/OpenPVSG/data/vidor"
    pvsg_json = "/scratch/saali/Datasets/OpenPVSG/data/pvsg.json"
    storage_dir = "/scratch/saali/Datasets/pvsg_results/vidor"
    num_workers = 4
    
    # dataset = EpicKitchenDataset(root_dir=root_dir, storage_dir=storage_dir, pvsg_json=pvsg_json, num_workers=num_workers, need_frames=True, need_jsons=False)
    dataset = VidOrDataset(root_dir=root_dir, storage_dir=storage_dir, pvsg_json=pvsg_json, num_workers=num_workers)


if __name__ == "__main__":
    main()
