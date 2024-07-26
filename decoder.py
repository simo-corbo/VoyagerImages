import numpy as np
import matplotlib.pyplot as plt
import scipy

def load_data():
    # load the audio file
    sample_rate, audio_data = scipy.io.wavfile.read('resources/voyager_images_384khz.wav')
    print(f"Sample rate: {sample_rate}")
    return audio_data[:, 0], audio_data[:, 1], sample_rate

def removeBeginning(channel, sample_rate, approximateEnd):
    audio_data_15s = channel[:round(15.45 * sample_rate)]

    # normalise the audio data
    audio_data_15s = audio_data_15s / np.max(audio_data_15s)

    # compute the peaks of the signal
    initialPeaks,_ = scipy.signal.find_peaks(audio_data_15s, height=np.max(audio_data_15s)*0.5, distance=sample_rate//10)


    return channel[initialPeaks[-1]:]

def extractImage(channel, sample_rate):
    imgs, _ = scipy.signal.find_peaks(channel, height=np.max(channel)*0.70, distance=sample_rate*5)
    imgs=np.append(imgs, len(channel)-1)
    return imgs

def fineTrim(channel, start, end, startFraction, endFraction, thresholdStart, thresholdEnd, height):
    full_image=channel[start:end]

    # normalise the image
    full_image = full_image / np.max(full_image)
    image=full_image
    
    
    # to find the effective length of the image, we need to trim the eventual image delimiters at the beginning and at the end

    # find the peaks in the first part of the image, to trim the beginning
    initalPeaks, _ = scipy.signal.find_peaks(image[:len(image)//startFraction], height=np.max(image)*height, threshold=thresholdStart)
    # trim the beginning of the image
    if len(initalPeaks) > 0:
        image = image[initalPeaks[-1]:]

    # now we need to trim the end of the image
    # find the peaks in the last part of the image
    peaks, _ = scipy.signal.find_peaks(image[-len(image)//endFraction:], height=np.max(image)*height, threshold=thresholdEnd)
    # trim the end of the image
    if len(peaks) > 0:
        image = image[:peaks[0]+image.size-len(image)//8]
    
    return image

def offsetDirection(peaks, width, height, img):
    cols = []

    for i in range(len(peaks)-1):
        col = img[peaks[i]:peaks[i+1]]
        col = 1-col
        if len(col)>0:
            col = scipy.signal.resample(col, width)
            cols.append(col)
    
    diffs = []
    #for i in range(0, width-1, 2):
    for i in range(0, len(cols)-1, 2):
        dip_i = scipy.signal.find_peaks(-cols[i], prominence=0.1, threshold=-1)
        dip_i1 = scipy.signal.find_peaks(-cols[i+1], prominence=0.1, threshold=-1)
        try:
            if len(dip_i[0]) > 0 and len(dip_i1[0]) > 0:
                diffs.append(dip_i[0][0]-dip_i1[0][0])
        except:
            # plot the dip_i and dip_i1
            plt.plot(-cols[i])
            plt.plot(-cols[i+1])
            plt.plot(dip_i[0][0], -cols[i][dip_i[0][0]], 'x')
            plt.plot(dip_i1[0][0], -cols[i+1][dip_i1[0][0]], 'x')
            plt.ylabel('Amplitude')
            plt.xlabel('Time [samples]')
            plt.title('Rows '+str(i)+', '+str(i+1)+' of the image')
            plt.legend(['Row '+str(i), 'Row '+str(i+1), 'Dip in row '+str(i), 'Dip in row '+str(i+1) ])
            plt.show()

            pass

    onEven = True
    if np.mean(diffs) < 0:
        onEven = False

    #print(f"Average: {round(np.mean(diffs))}")
        
    return onEven


def developImage(img, sample_rate, offset):
    # now we can find the column delimiters as the peaks the trimmed image
    peaks, _ = scipy.signal.find_peaks(img, distance=sample_rate/200)
    # each peak is a column delimiter, we can now extract the columns and plot them in a canvas
    cols = []
    # the height of the column is based on the proportion of the image
    width=512
    # ratio between the width and the height of the image is 4:3
    height=round(width*3/4)


    onEven= offsetDirection(peaks, width, height, img)
    # extract the columns
    #for i in range(width-1):
    for i in range(len(peaks)-1):
        # extract the column
        col = img[peaks[i]:peaks[i+1]]
        # an higher value of the column corresponds to a black pixel, so we need to invert the column
        #col = 1-col
        
        # on even columns, remove the first 10 samples and add 10 duplicates of the last sample


        direction = 0 if onEven else 1
        if i%2==direction:
            col = col[offset:]
        else:
            col = np.concatenate([col, np.full(offset, col[-1])])
              
        col = scipy.signal.resample(col, height)
        # append the column to the rows
        cols.append(col)

    # pick the last width columns
    #cols = cols[-width:]
    # transpose the columns to have the image in the right orientation
    img = np.array(cols).T
    # Get upper and lower percentiles of image data.
    low = np.percentile(img, 2)
    high = np.percentile(img, 98)
    
    # Normalise image contrast and invert image.
    img_data = np.clip(img, low, high)
    img_data = 255 - ((img - low) / (high - low)) * 255

    return img_data

def computeOffset(img, height, sample_rate):
    colPeaks, _ = scipy.signal.find_peaks(img, distance=sample_rate/200, height=np.max(img)*0.3)
    cols=[]
    for i in range(len(colPeaks)-1):
        # extract the column
        col = img[colPeaks[i]:colPeaks[i+1]]
        # an higher value of the column corresponds to a black pixel, so we need to invert the column
        col = 1-col

        # the column is resized to the height of the image*10
        col = scipy.signal.resample(col, height*10)
        # append the column to the rows
        cols.append(col)
    diffs = []
    for i in range(0, len(cols)-1, 2):
        dip_i, _ = scipy.signal.find_peaks(cols[i], prominence=0.30, threshold=-1)
        dip_i1, _ = scipy.signal.find_peaks(cols[i+1], prominence=0.30, threshold=-1)
        if len(dip_i) > 0 and len(dip_i1) > 0:
            # append the difference in samples between the dips
            diffs.append(dip_i1[0]-dip_i[0])
    # average
    print(f"Average: {round(np.mean(diffs))}")
    return abs(round(np.mean(diffs)))

def main():
    # load the audio data
    left_channel, right_channel, sample_rate = load_data()
    left_colour_indexes=[[7, 8, 9], [13, 14, 15], [16, 17, 18], [28, 29, 30], [41, 42, 43], [44, 45, 46], [47, 48, 49], [58, 59, 60], [61, 62, 63], [65, 66, 67], [68, 69, 70], [71, 72, 73]]
    right_colour_indexes=[[0, 1, 2], [7, 8, 9], [27, 28, 29], [40, 41, 42], [47, 48, 49], [52, 53, 54], [69, 70, 71], [73, 74, 75]]

    channels = [(left_channel, left_colour_indexes), (right_channel, right_colour_indexes)]


    # the height of the column is based on the proportion of the image
    width=512
    # ratio between the width and the height of the image is 4:3
    height=round(width*3/4)
    imageCounter=0

    # computed with the first image
    offsetAmount=0

    for channel, color_index in channels:
        # remove the beginning of the audio data
        channel = removeBeginning(channel, sample_rate, 15.45)
        
        # extract the images from the audio data
        imgs = extractImage(channel, sample_rate)

        decoded_imgs = []

        for i in range(0, len(imgs)-1):
            # empirical values for the trimming of the image
            img = fineTrim(channel, imgs[i], imgs[i+1],  20, 20, 0, 0, 0.7)
            # 100 is the offset, got empirically
            if i == 0:
                offsetAmount = computeOffset(img, height, sample_rate)
            img= developImage(img, sample_rate, offsetAmount)
            decoded_imgs.append(img)
        
        black_white_imgs = [i for i in range(0, len(decoded_imgs))]
        for image in color_index:
            for i in image:
                black_white_imgs.remove(i)
        
        for colors in color_index:
            # Get RGB channels for color image.
            r, g, b = [decoded_imgs[c] for c in colors]

            # Crop image to smallest sized channel.
            edge = min(max(r.shape), max(g.shape), max(b.shape))
            r, g, b = (r[:,:edge], g[:,:edge], b[:,:edge])

            # Combine color channels into a single image.
            color_img = np.dstack((r,g,b))

            # normalise the image to 0-255 to integers
            color_img = (color_img - np.min(color_img)) / (np.max(color_img) - np.min(color_img))
            
        # create the subplots, as many as the number of black and white and coloured images
        n_cols = 6
        number_of_images = len(black_white_imgs) + len(color_index)
        n_rows = number_of_images // n_cols + 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 30))

        # plot the black and white images
        for i in range(len(black_white_imgs)):
            img = decoded_imgs[black_white_imgs[i]]
            # save the image
            plt.imsave(f'output/image_{imageCounter}.png', img, cmap='gray')
            axs.flat[i].imshow(img, cmap='gray')
            axs.flat[i].axis('off')
            axs.flat[i].set_title(f'Image {black_white_imgs[i]}')
            imageCounter+=1
            

        # plot the coloured images
        for i in range(len(color_index)):
            colors = color_index[i]
            r, g, b = [decoded_imgs[c] for c in colors]
            edge = min(max(r.shape), max(g.shape), max(b.shape))
            r, g, b = (r[:,:edge], g[:,:edge], b[:,:edge])
            color_img = np.dstack((r,g,b))
            color_img = (color_img - np.min(color_img)) / (np.max(color_img) - np.min(color_img))
            # save the image
            plt.imsave(f'output/image_{imageCounter}.png', color_img)
            imageCounter+=1
            axs.flat[i+len(black_white_imgs)].imshow(color_img)
            axs.flat[i+len(black_white_imgs)].axis('off')
            axs.flat[i+len(black_white_imgs)].set_title(f'Image {colors[0]}-{colors[-1]}')


        plt.show()




if __name__ == '__main__':
    main()