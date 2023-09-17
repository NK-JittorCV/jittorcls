import argparse
import numpy as np
from tqdm import tqdm
import jittor as jt
from jittor import transform
from jvcore.models import create_model
from jvcore.data import create_dataset
jt.flags.use_cuda = 1



def get_valid_transforms():
    return transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


@jt.no_grad()
@jt.single_process_scope()
def val(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    pbar = tqdm(val_loader, desc=f'[VALID]', maxinterval=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        # total_acc = jt.int64(total_acc)
        # total_num = jt.int64(total_num)
        pbar.update(1)
        pbar.set_description(f'Test Epoch: {epoch} [{batch_idx}/{len(val_loader)}]\tAcc: {acc:.6f}')
    # if jt.in_mpi:
    #     total_acc = total_acc.mpi_all_reduce()
    #     total_num = total_num.mpi_all_reduce()
    # if jt.rank == 0:
    #     return total_acc.numpy() / total_num.numpy()
    return total_acc / total_num
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='p2t_tiny')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    val_loader = create_dataset(args.dataset, root=args.data_root, split='validation', transform=get_valid_transforms(), 
                                batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    model = create_model(args.model, pretrained=True, checkpoint_path=args.checkpoint_path)

    acc = val(model, val_loader, 1)
    print('Final val acc:', acc)
