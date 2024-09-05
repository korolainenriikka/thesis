import pandas as pd
import cv2
from argparse import ArgumentParser
import os
import json

def read_json_sentences_to_df(json_path: str) -> pd.DataFrame:
    """Get pd.DataFrame with bounding box details derived from given JSON file.

    The dataframe has the following columns:
    'text',
    'x_ul', 'x_ur', 'x_ll', 'x_lr',
    'y_ul', 'y_ur', 'y_ll', 'y_lr',
    'x_center', 'y_center'.

    :param json_path:
    :return: pd.DataFrame:
    """
    with open(json_path) as f:
        json_obj = json.load(f)
    x_upper_left = []
    x_upper_right = []
    x_lower_left = []
    x_lower_right = []

    y_upper_left = []
    y_upper_right = []
    y_lower_left = []
    y_lower_right = []

    text = []

    for box in json_obj['readResult']['blocks'][0]['lines']:
        bp = box['boundingPolygon']
        x_upper_left.append(bp[3]['x'])
        x_upper_right.append(bp[2]['x'])
        x_lower_left.append(bp[0]['x'])
        x_lower_right.append(bp[1]['x'])

        y_upper_left.append(bp[3]['y'])
        y_upper_right.append(bp[2]['y'])
        y_lower_left.append(bp[0]['y'])
        y_lower_right.append(bp[1]['y'])

        text.append(box['text'])

    box_df = pd.DataFrame(
        {
            'text': text,
            'x_ul': x_upper_left,
            'x_ur': x_upper_right,
            'x_ll': x_lower_left,
            'x_lr': x_lower_right,

            'y_ul': y_upper_left,
            'y_ur': y_upper_right,
            'y_ll': y_lower_left,
            'y_lr': y_lower_right
        }
    )
    box_df['x_center'] = (box_df['x_ul'] + box_df['x_ur'] + box_df['x_ll'] + box_df['x_lr']) / 4
    box_df['y_center'] = (box_df['y_ul'] + box_df['y_ur'] + box_df['y_ll'] + box_df['y_lr']) / 4

    return box_df

def main(directory: str, final_data_destination: str):
    print('Starting card processing...')

    # Find paths for all subdirectories containing one accession
    paths_with_accession_data = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
               paths_with_accession_data.append(root)
               break

    # Constants
    card1_png  = 'card1.png'
    card1_json = 'card1.json'

    i = 0
    for accession_dir in paths_with_accession_data:
        try:
            print(f'*** Found accession data in {accession_dir}, processing... ***')

            json_path = os.path.join(accession_dir, card1_json)
            card1_df = read_json_sentences_to_df(json_path)
            element_row_id = card1_df.loc[card1_df['text'].str.contains('NATURE OF SPECIMEN')].index + 1
            element_row = card1_df.loc[element_row_id]
            
            img_path = os.path.join(accession_dir, card1_png)
            img = cv2.imread(img_path)
           
            y1 = min(int(element_row['y_ur']), int(element_row['y_ul']))
            y2 = max(int(element_row['y_lr']), int(element_row['y_ll']))
            # -10 + 10 to get a bit larger section
            y_top = min(y1, y2) - 10
            y_bottom = max(y1, y2) + 10

            x1 = min(int(element_row['x_ll']), int(element_row['x_ul']))
            x2 = max(int(element_row['x_lr']), int(element_row['x_ur']))
            x_left = min(x1, x2) - 10
            x_right = max(x1, x2) + 10

            crop_img = img[y_top:y_bottom,x_left:x_right]
            dest_filename = os.path.join(final_data_destination, str(i) + '.png')
            cv2.imwrite(dest_filename, crop_img)
            i += 1

        except Exception as e:
            print(f'ERROR: {e}')
            print('Unexpected error encountered, skipping accession')
        print()

    print('*** Finished ***')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('directory', help='Path to directory where cards are stored')
    parser.add_argument('final_data_destination', help='Path to directory where cropped images will be saved')
    args = parser.parse_args()
    main(directory=args.directory, final_data_destination=args.final_data_destination)
