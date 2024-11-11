import json 
import re
import cv2
import argparse
import os

def bounding_box_to_segment(image, bb):
    x_coords = [point['x'] for point in bb]
    y_coords = [point['y'] for point in bb]

    # Determine the bounding box
    left = min(x_coords)
    top = min(y_coords)
    right = max(x_coords)
    bottom = max(y_coords)
    return image[top:bottom, left:right]

def extract_data(source_catalogue, destination_directory):
    # intentionally crash if dir exists to not overwrite old data
    os.mkdir(destination_directory)
    teeth_found = 0
    azure_labels = []
    img_i = 0
    for i in range(1,1000): # fallback
        file_path = f'{source_catalogue}_pages/page{i+1}.json'
        pdf = cv2.imread(f'{source_catalogue}_pages/page{i+1}.png')

        try:
            with open(file_path, 'r') as json_file:
                # Load the JSON data into a Python object
                data = json.load(json_file)
        except FileNotFoundError:
            # ran out of pages
            print(f'* Finished *')
            print(f'Teeth found: {teeth_found}')
            break

        for block in data['readResult']['blocks']:
            for line in block['lines']:
                for i, word in enumerate(line['words']):
                    is_tooth = re.match('^[a-zA-Z]\d$|^[cC]$', word['text'])
                    if (is_tooth):
                        teeth_found += 1
                        azure_label = word['text']
                        bounding_box = word['boundingPolygon']
                        azure_labels.append(azure_label)
                        img_segment = bounding_box_to_segment(pdf, bounding_box)
                        cv2.imwrite(f'{destination_directory}/{img_i}.png', img_segment)
                        img_i += 1
    
    with open(f'{destination_directory}/azure_labels.txt', 'w') as out:
        out.write('\n'.join(azure_labels))

    with open(f'{destination_directory}/about.txt', 'w') as about:
        about.write(f'Source: {source_catalogue}, no. of dental markings found: {teeth_found}')

    

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process source catalogue and destination directory paths.")
    
    # Add arguments for source catalogue and destination directory
    parser.add_argument(
        '--source_catalogue', 
        type=str, 
        required=True, 
        help="Path to the source catalogue file."
    )
    
    parser.add_argument(
        '--destination_directory', 
        type=str, 
        required=True, 
        help="Path to the destination directory."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print or process the arguments
    extract_data(args.source_catalogue, args.destination_directory)

if __name__ == "__main__":
    main()