from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='cd1.4',
                    
                       captions_per_image=1,
                       min_word_freq=5,
                       output_folder=r'/Users/pranithred/Desktop/clef23/captioning/caption_data',

                       )