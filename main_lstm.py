import keras_tools.lstm as lstm

folder = "/media/winbuntu/google-drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/of_features"
# folder = "/media/sutadasuto/OS/Users/Sutadasuto/Google Drive/INAOE/Thesis/Real-life_Deception_Detection_2016/Clips_/covarep_features"

# lstm.test()
# my_lstm = lstm.basic_binary_lstm_cv(folder)
lstm.standard_vs_binary(folder)
