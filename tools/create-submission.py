import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import tqdm

from contest.segmentation.models import get_model as get_segmentation_model
from contest.segmentation.models import get_model1, get_model2, get_model3, get_model4, get_model5, get_model6
from contest.recognition.model import get_model as get_recognition_model
from contest.common import prepare_for_segmentation, get_boxes_from_mask, prepare_for_recognition


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None, help="path to the data")
    parser.add_argument("-s1", "--seg-model1", dest="seg_model1", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s2", "--seg-model2", dest="seg_model2", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s3", "--seg-model3", dest="seg_model3", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s4", "--seg-model4", dest="seg_model4", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s5", "--seg-model5", dest="seg_model5", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s6", "--seg-model6", dest="seg_model6", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s7", "--seg-model7", dest="seg_model7", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-s", "--seg-model", dest="seg_model", type=str, default=None,
                        help="path to a trained segmentation model")
    parser.add_argument("-r", "--rec-model", dest="rec_model", type=str, default=None,
                        help="path to a trained recognition model")
    parser.add_argument("-o", "--output_file", dest="output_file", default="baseline_submission.csv",
                        help="file to save predictions to")
    return parser.parse_args()


def main(args):
    print("Start inference")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### single model prediction
    # segmentation_model = get_segmentation_model()
    # with open(args.seg_model, "rb") as fp:
    #     state_dict = torch.load(fp, map_location="cpu")
    # segmentation_model.load_state_dict(state_dict)
    # segmentation_model.to(device)
    # segmentation_model.eval()
    
    segmentation_model1 = get_model1()
    with open(args.seg_model1, "rb") as fp:
        state_dict1 = torch.load(fp, map_location="cpu")
    segmentation_model1.load_state_dict(state_dict1)
    segmentation_model1.to(device)
    segmentation_model1.eval()
    
    segmentation_model2 = get_model2()
    with open(args.seg_model2, "rb") as fp:
        state_dict2 = torch.load(fp, map_location="cpu")
    segmentation_model2.load_state_dict(state_dict2)
    segmentation_model2.to(device)
    segmentation_model2.eval()
    
    segmentation_model3 = get_model3()
    with open(args.seg_model3, "rb") as fp:
        state_dict3 = torch.load(fp, map_location="cpu")
    segmentation_model3.load_state_dict(state_dict3)
    segmentation_model3.to(device)
    segmentation_model3.eval()
    
    segmentation_model4 = get_model4()
    with open(args.seg_model4, "rb") as fp:
        state_dict4 = torch.load(fp, map_location="cpu")
    segmentation_model4.load_state_dict(state_dict4)
    segmentation_model4.to(device)
    segmentation_model4.eval()
    
    segmentation_model5 = get_model5()
    with open(args.seg_model5, "rb") as fp:
        state_dict5 = torch.load(fp, map_location="cpu")
    segmentation_model5.load_state_dict(state_dict5)
    segmentation_model5.to(device)
    segmentation_model5.eval()
    
    
    segmentation_model6 = get_model6()
    with open(args.seg_model6, "rb") as fp:
        state_dict6 = torch.load(fp, map_location="cpu")
    segmentation_model6.load_state_dict(state_dict6)
    segmentation_model6.to(device)
    segmentation_model6.eval()
    
    
    recognition_model = get_recognition_model()
    with open(args.rec_model, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    recognition_model.load_state_dict(state_dict)
    recognition_model.to(device)
    recognition_model.eval()

    
    test_images_dirname = os.path.join(args.data_path, "test")
    results = []
    files = os.listdir(test_images_dirname)
    for i, file_name in enumerate(tqdm.tqdm(files)):
        image_src = cv2.imread(os.path.join(test_images_dirname, file_name))
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        
        # image, (kx, ky) = prepare_for_segmentation(image_src.astype(np.float32) / 255., (512, 512))
        # x = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        # with torch.no_grad():
        #     pred = torch.sigmoid(segmentation_model(x.to(device))).squeeze().cpu().numpy()
        # mask = (pred >= 0.5).astype(np.uint8) * 255
        
        # # 1. Segmentation.
        image, (kx, ky) = prepare_for_segmentation(image_src.astype(np.float32) / 255., (512, 512))
        x = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        with torch.no_grad():
            pred1 = torch.sigmoid(segmentation_model1(x.to(device))).squeeze().cpu().numpy()
            pred2 = torch.sigmoid(segmentation_model2(x.to(device))).squeeze().cpu().numpy()
            #pred3 = torch.sigmoid(segmentation_model3(x.to(device))).squeeze().cpu().numpy()
            #pred4 = torch.sigmoid(segmentation_model4(x.to(device))).squeeze().cpu().numpy()
            #pred5 = torch.sigmoid(segmentation_model5(x.to(device))).squeeze().cpu().numpy()
            #pred6 = torch.sigmoid(segmentation_model6(x.to(device))).squeeze().cpu().numpy()
        pred = (pred1 + pred2) / 2
        mask = (pred >= 0.5).astype(np.uint8) * 255

        # 2. Extraction of detected regions.
        boxes = get_boxes_from_mask(mask)
        if len(boxes) == 0:
            results.append((file_name, []))
            continue
        boxes = boxes.astype(np.float32)
        boxes[:, [0, 2]] *= kx
        boxes[:, [1, 3]] *= ky
        boxes = boxes.astype(np.int32)

        # 3. Text recognition for every detected bbox.
        texts = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                raise (Exception, str(box))
            if (y2 - y1) < 3 or (x2 - x1) < 3:  # skip small boxes
                continue
            crop = image_src[y1: y2, x1: x2, :]

            tensor = prepare_for_recognition(crop, (640, 128)).to(device)
            with torch.no_grad():
                text = recognition_model(tensor, decode=True)[0]
            texts.append((x1, text))

        # all predictions must be sorted by x1
        texts.sort(key=lambda x: x[0])
        results.append((file_name, [w[1] for w in texts]))

    # Generate a submission file
    with open(args.output_file, "wt") as wf:
        wf.write("file_name,plates_string\n")
        for file_name, texts in sorted(results, key=lambda x: int(os.path.splitext(x[0])[0])):
            wf.write(f"test/{file_name},{' '.join(texts)}\n")
    print('Done')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
