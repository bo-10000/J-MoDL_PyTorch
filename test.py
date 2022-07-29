import argparse
import yaml, os, time
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

from get_instances import *
from utils import *

def setup(args):
    config_path = args.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = 'cuda'

    #read configs =================================
    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']

    batch_size = configs['batch_size'] if args.batch_size is None else args.batch_size

    model_name = configs['model_name']
    model_params = configs.get('model_params', {})
    model_params['initx'], model_params['inity'] = get_init_mask(configs['init_mask_path'])

    score_names = configs['score_names']

    config_name = configs['config_name']

    workspace = os.path.join(args.workspace, config_name) #workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace) #workspace/config_name/checkpoints ; workspace/config_name/log.txt
    tensorboard_dir = os.path.join(args.tensorboard_dir, configs['config_name']) #runs/config_name
    logger = Logger(log_dir)
    writer = get_writers(tensorboard_dir, ['test'])['test']

    dataloader = get_loaders(dataset_name, dataset_params, batch_size, ['test'])['test']
    model = get_model(model_name, model_params, device)
    score_fs = get_score_fs(score_names)

    #restore
    saver = CheckpointSaver(checkpoints_dir)
    prefix = 'best' if configs['val_data'] else 'final'
    checkpoint_path = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith(prefix)][0]
    model = saver.load_model(checkpoint_path, model)

    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model)

    return configs, device, workspace, logger, writer, dataloader, model, score_fs

def main(args):
    start = time.time()
    set_seeds(args.seed) #for random noise
    
    configs, device, workspace, logger, writer, dataloader, model, score_fs = setup(args)

    logger.write('\n')
    logger.write('test start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    running_score = defaultdict(int)

    model.eval()
    for i, (gt, csm) in enumerate(tqdm(dataloader)):
        gt, csm = gt.to(device), csm.to(device)

        with torch.no_grad():
            x_u, x_rec = model(gt.clone(), csm)

        gt = np.abs(gt.detach().cpu().numpy())
        x_u = np.abs(x_u.detach().cpu().numpy())
        x_rec = np.abs(x_rec.detach().cpu().numpy())
        
        for score_name, score_f in score_fs.items():
            running_score[score_name] += score_f(gt, x_rec) * x_rec.shape[0]
        if args.write_image > 0 and (i % args.write_image == 0):
            mask = get_mask_img(model.at.kx.detach().cpu(), model.at.ky.detach().cpu(), *x_rec.shape[-2:])
            writer.add_figure('img', display_img(x_u[-1], mask, gt[-1], x_rec[-1], psnr(gt[-1], x_rec[-1], gt[-1].max())), i)

    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        writer.add_scalar(score_name, score, 0)
        logger.write('test {} score: {:.4f}'.format(score_name, score))

    writer.close()
    logger.write('-----------------------')
    logger.write('total test time: {:.2f} min'.format((time.time()-start)/60))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/base_modl,k=1.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--write_image", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    main(args)
