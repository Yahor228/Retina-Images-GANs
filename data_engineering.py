import PIL
import PIL.Image
import os
import shutil 
import cv2
import numpy as np 

from patchify import patchify , unpatchify


def make_dataset(absolute_path, answers):

    paths = ['Healthy', "Mild", "Moderate", "Severe", "Proliferative DR" ]

    def get_key(value):
        a = answers[answers[value]]
        if a != None:
            return a["level"]
        
    files = os.listdir(absolute_path)
    for file in files:
        extension = file.split(".")
        obj =  answers[answers["image"] == extension[0]]
        a = obj["level"].tolist()
        b = obj["image"]
        if len(a) == 0:
            continue
        shutil.move(absolute_path + file.rstrip("\r\n"), absolute_path + paths[a[0]] + "\\" + file.rstrip("\r\n"))



def countImages(folder):
    in_folder = folder

    file_count = []

    # get number of images in each folder (images per class)
    for fld in os.listdir(in_folder):
        crt = os.path.join(in_folder, fld)
        
        image_count = len(os.listdir(crt))
        
        file_count.append((fld, image_count))
        
        print(f'{crt} contains {image_count} images')

    return file_count





#очистить траин и тест папки , создать подпапки с нужными классами , расскидать по папкам 
def split(folder, train_folder, test_folder, coef):


    classes = countImages(folder)

    for f in [test_folder,train_folder]:
        for filename in os.listdir(f):
            file_path = os.path.join(f, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


    min_count = min([i[1] for i in classes])
    train_count = int(min_count * (1 - coef)) 
    test_count = min_count - train_count
    for c,count in classes:
        print("sorting", count, "images for class", c)
        print("overall:", min_count, "train/test set:", train_count, test_count)
        for f in [test_folder,train_folder]:
            counter = 0
            os.mkdir(f + "/" + c)
        for file in os.listdir(folder + '/' + c):
            counter += 1
            if counter <= train_count:
                dst = train_folder + "/" + c 
            else:
                dst = test_folder + "/" + c 


            shutil.copy(folder + "/" + c + "/"+ file, dst)




def makemirror(img, tw, th, ov):
    img = PIL.Image.open(img)
    img = np.asarray(img)
    
    # Check if the input image is grayscale (has only one channel)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)  # Add a third dimension to represent the single channel
    
    # Get the size of the input image
    h, w, channels = img.shape  # channels will be 1 for grayscale and 3 for RGB
    
    # Compute the number of blocks in the horizontal and vertical directions
    n_blocks_w = (w // tw) + (0 if w % tw == 0 else 1)
    n_blocks_h = (h // th) + (0 if h % th == 0 else 1)
    
    # Compute the size of the output image
    out_h = n_blocks_h * th + 2 * ov
    out_w = n_blocks_w * tw + 2 * ov
    
    # Create an array to hold the output image
    out_img = np.zeros((out_h, out_w, channels), dtype=img.dtype)
    
    # Compute the indices of the blocks in the output image
    block_indices_w = np.arange(n_blocks_w) * tw
    block_indices_h = np.arange(n_blocks_h) * th
    
    # Copy the blocks from the input image into the output image
    for i, x in enumerate(block_indices_w):
        for j, y in enumerate(block_indices_h):
            block = img[max(y - ov, 0):min(y + th + ov, h), max(x - ov, 0):min(x + tw + ov, w)]
            block_h, block_w = block.shape[:2]
            out_img[j*th+ov:j*th+ov+block_h, i*tw+ov:i*tw+ov+block_w] = block
    
    # Fill the borders with mirrored pixels from the input image
    out_img[:ov, ov:out_w-ov] = np.flipud(out_img[ov:2*ov, ov:out_w-ov])
    out_img[out_h-ov:, ov:out_w-ov] = np.flipud(out_img[out_h-2*ov:out_h-ov, ov:out_w-ov])
    out_img[ov:out_h-ov, :ov] = np.fliplr(out_img[ov:out_h-ov, ov:2*ov])
    out_img[ov:out_h-ov, out_w-ov:] = np.fliplr(out_img[ov:out_h-ov, out_w-2*ov:out_w-ov])
    out_img[:ov, :ov] = np.fliplr(np.flipud(img[:ov, :ov]))
    out_img[:ov, out_w-ov:] = np.fliplr(np.flipud(img[:ov, w-ov:]))
    out_img[out_h-ov:, :ov] = np.fliplr(np.flipud(img[h-ov:, :ov]))
    out_img[out_h-ov:, out_w-ov:] = np.fliplr(np.flipud(img[h-ov:, w-ov:]))
    
    return out_img




# def crop(img, th, tw, ov):
    
#     height, width = img.shape[:2]
    
    
#     # Calculate the number of pieces in each dimension
#     num_h_pieces =  (height - ov) // th 
#     num_w_pieces =  (width - ov) // tw 
    
#     # Create an array to hold the cropped pieces
#     cropped_pieces = np.zeros((num_h_pieces * num_w_pieces, th, tw, 3), dtype=img.dtype)

#     # Loop through each piece and crop it from the input image      
    
#     for h_idx in range(num_h_pieces):
#         for w_idx in range(num_w_pieces):
#             h_start = h_idx * th - ov
#             h_end = h_start + th + ov
#             w_start = w_idx * tw - ov
#             w_end = w_start + tw + ov 
#             cropped_pieces[h_idx * num_w_pieces + w_idx, :h_end - h_start, :w_end - w_start] = img[h_start:h_end, w_start:w_end]

#     return cropped_pieces


def crop(img, th, tw, ov):
    height, width = img.shape[:2]
    num_h_pieces = (height - ov) // th 
    num_w_pieces = (width - ov) // tw 
    cropped_pieces = np.zeros((num_h_pieces * num_w_pieces, th, tw, 3), dtype=img.dtype)

    for h_idx in range(num_h_pieces):
        for w_idx in range(num_w_pieces):
            h_start = h_idx * (th - ov)
            h_end = h_start + th
            w_start = w_idx * (tw - ov)
            w_end = w_start + tw
            cropped_pieces[h_idx * num_w_pieces + w_idx, :, :, :] = img[h_start:h_end, w_start:w_end, :]

    return cropped_pieces


def grayscale_cropped_pieces(cropped_pieces):
    num_pieces = cropped_pieces.shape[0]
    grayscale_pieces = np.zeros((num_pieces, cropped_pieces.shape[1], cropped_pieces.shape[2], 1), dtype=cropped_pieces.dtype)

    for idx in range(num_pieces):
        piece = cropped_pieces[idx]
        grayscale_piece = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
        grayscale_pieces[idx, :, :, 0] = grayscale_piece

    return grayscale_pieces



def save_cropped_pieces(cropped_pieces, output_dir):
    """
    Сохраняет каждый обрезанный кусок как изображение в формате jpg.

    Args:
    cropped_pieces (numpy.ndarray): Массив обрезанных кусков с формой (количество_кусков, высота, ширина, каналы).
    output_dir (str): Директория, в которой будут сохранены обрезанные изображения.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, piece in enumerate(cropped_pieces):
        if piece.shape[-1] == 1:  # Изображение в оттенках серого
            piece = np.squeeze(piece, axis=-1)  # Удаляем последнее измерение
            image = PIL.Image.fromarray(piece, mode="L")  # Используем режим "L" для оттенков серого
        else:
            image = PIL.Image.fromarray(piece)

        image_path = os.path.join(output_dir, f"{i}.jpg")
        image.save(image_path)







def walk_by_pic(root_folder):

    os.mkdir("patches")
    for root, dirs, files in os.walk(root_folder):
            for file in files:
                # Check if file is an image file
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    filepath = root + '/' + file
                
                    a =  crop(filepath, 128, 128, 32,"patches")





def convert_images_to_arrays(directory, buffer_size=100):
    image_arrays = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
                path = os.path.join(root, file)
                image = PIL.Image.open(path)
                array = np.asarray(image)
                image_arrays.append(array)
                if len(image_arrays) == buffer_size:
                    np.save('arrays.npy', np.array(image_arrays))
                    image_arrays = []
    if len(image_arrays) > 0:
        np.save('arrays.npy', np.array(image_arrays))
        

def process_images(input_folder, output_folder, is_a_pic=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Получаем список файлов в папке с исходными изображениями
    files = os.listdir(input_folder)

    for file in files:
        # Проверяем, является ли файл изображением с поддерживаемым расширением (например, jpg, tif)
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            input_img_path = os.path.join(input_folder, file)

            # Формируем префикс имени файла для сохранения
            output_img_prefix = os.path.join(output_folder, f'{os.path.splitext(file)[0]}')  # Убираем расширение

            # Применяем функции обработки изображения
            img = makemirror(input_img_path, tw=512, th=512, ov=32)
            cropped_pieces = crop(img, th=512, tw=512, ov=32)
            if is_a_pic:
                cropped_pieces = grayscale_cropped_pieces(cropped_pieces)

            # Удаляем измерение с одним каналом, если оно существует
            if cropped_pieces.shape[-1] == 1:
                cropped_pieces = cropped_pieces.squeeze(axis=-1)

            # Сохраняем обработанные изображения с добавлением номера части в имя файла
            for idx, piece in enumerate(cropped_pieces):
                piece_img = PIL.Image.fromarray(piece)
                if is_a_pic:
                    piece_img = piece_img.convert("L")  # Конвертируем в оттенки серого
                piece_img.save(f'{output_img_prefix}_{idx+1:02d}.jpg')