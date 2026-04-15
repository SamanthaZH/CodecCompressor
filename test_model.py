import argparse
from datetime import datetime
import gin
from loguru import logger
from torch.utils.data import DataLoader

from utils.common import set_random_seed
from dataset.ray_dataset import RayDataset, ray_collate
from neural_field.model import get_model
from trainer import Trainer
import torch 

@gin.configurable()
def main(
    seed: int = 42,
    num_workers: int = 0,
    train_split: str = "train",
    stages: str = "eval",
    batch_size: int = 16,
    model_name="Tri-MipRF",
    saved_model_dir = "",
):
    stages = list(stages.split("_"))
    set_random_seed(seed)

    logger.info("==> Init dataloader ...")

    test_dataset = RayDataset(split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=None,
        pin_memory_device='cuda',
    )

    logger.info("==> Init model ...")
    model = get_model(model_name=model_name)(aabb=test_dataset.aabb)
    logger.info(model)

    logger.info("==> Init trainer ...")
    trainer = Trainer(model, None, eval_loader=test_loader, test_chunk_size = 8192*4, num_rays = 8192*4)
    
    if "eval" in stages:
        '''
        for name in ['ckpt','mlp']:
            for idx in ['0','10','20','30','40']:
                nm = name + idx +'.ckpt'
                print("===> current ckpt: " + nm)
                trainer.load_ckpt_test(nm)
                print('==> trainer_ckpt:',trainer.load_ckpt_test(nm))
                trainer.eval(save_results=True, rendering_channels=["rgb", "depth"])
        '''
        trainer.load_ckpt()
        print('==> trainer_ckpt:',trainer.load_ckpt())

        trainer.eval(save_results=True, rendering_channels=["rgb", "depth"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    gin.parse_config_files_and_bindings(args.ginc, ginbs, finalize_config=False)

    exp_name = gin.query_parameter("Trainer.exp_name")
    exp_name = (
        "%s/%s/%s/%s"
        % (
            gin.query_parameter("RayDataset.scene_type"),
            gin.query_parameter("RayDataset.scene"),
            gin.query_parameter("main.model_name"),
            #datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            #"model_w_ds",
            gin.query_parameter("main.saved_model_dir"),
        )
        if exp_name is None
        else exp_name
    )
    print(exp_name)
    gin.bind_parameter("Trainer.exp_name", exp_name)
    gin.finalize()
    main()
