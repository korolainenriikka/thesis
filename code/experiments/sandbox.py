from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

def evaluate(generated, label):
    """
    Generate word counts required for precision and recall

    Args:
        generated: sentence generated with OCR
        label:     correct text in the image

    Returns:
        int: count of correct words in the generated sequence (correct word in wrong position is wrong)
        int: count of words in the generated sequence
        int: count of words in the label
    """
    generated_words = generated.split(' ')
    label_words = label.split(' ')
    correct_words = 0

    generated_word_count = len(generated_words)
    label_word_count = len(label_words)
    for i in range(generated_word_count):
        if i >= label_word_count:
            break
        if label_words[i] == generated_words[i]:
            correct_words += 1

    return correct_words, generated_word_count, label_word_count

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
path = '/home/riikoro/fossil_data/tooth_samples/v1'
with open(os.path.join(path, 'labels.txt')) as label_file:
    labels = label_file.readlines()

total_correct = 0
total_generated = 0
total_label_words = 0
for file in os.listdir(path):
    img_path = os.path.join(path, file)
    img_no = int(file.split('.')[0])

    with Image.open(img_path) as image:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        correct_label = labels[img_no]
        print('--- yks sana taas juuuu ---')
        print(generated_text)
        print(correct_label)
        print
        correct_word_count, generated_word_count, label_word_count = evaluate(generated_text, correct_label)
        print(correct_word_count)
        print(generated_word_count)
        print(label_word_count)

        total_correct += correct_word_count
        total_generated += generated_word_count
        total_label_words += label_word_count
