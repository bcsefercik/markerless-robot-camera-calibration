import os
import glob
import argparse
import random
import shutil
import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ycb")
    parser.add_argument("--infolder", type=str, default="ycb/")
    parser.add_argument("--outfolder", type=str, default="ycb_out/")
    parser.add_argument("--move", type=bool, default=False)
    parser.add_argument("--ratio", nargs="+", type=float, default=[0.85, 0.1, 0.05], help='train, val, test')
    args = parser.parse_args()

    class_folders = glob.glob(os.path.join(args.infolder, '*'))
    class_folders = [cf for cf in class_folders if os.path.isdir(cf)]
    data_types = [
        'train',
        'val',
        'test'
    ]

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    for i, dt in enumerate(data_types):
        destination = os.path.join(args.outfolder, dt)
        if not os.path.exists(destination):
            os.mkdir(destination)

    for cf in class_folders:
        print('Processing:', cf)
        pcd_files = glob.glob(os.path.join(cf, 'clouds', '*.pcd'))
        random.shuffle(pcd_files)

        findex = [int(r * len(pcd_files)) for r in args.ratio]
        findex.insert(0, 0)
        for i in range(1, len(findex)):
            findex[i] += findex[i - 1]
        findex[-1] = len(pcd_files)

        class_info = cf.split('/')[-1]

        for i, dt in enumerate(data_types):
            destination_folder = os.path.join(args.outfolder, dt)

            for fp in pcd_files[findex[i]:findex[i + 1]]:
                destination = os.path.join(
                    destination_folder,
                    f"{class_info}_{fp.split('/')[-1]}"
                )

                if args.move:
                    shutil.move(fp, destination)
                else:
                    shutil.copy(fp, destination)

        # ipdb.set_trace()
