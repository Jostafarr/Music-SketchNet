from preprocessing.irish_process import process_irish
from ec2vae.train import train_ec2vae
from MeasureVAE.train import train_measure_vae
from SketchVAE.data_process import process_data
from SketchVAE.train import train_sketch_vae
from music_inpaintNet import train_music_inpaintNet
from sketchVAE_inpaintRNN import train_sketchvae_inpaintRNN
from SketchNet.train import train_sketchnet
from musicInpaintNet_eval import evaluate_musicInpaintNet
from SketchVAE_InpaintRNN_eval import evaluata_SketchVAE_InpaintRNN
from SketchNet.evaluate import evaluate_sketchNet
from utils.loss_and_accuracy import get_loss_and_accuracy




s_dir = "" # folder address
dataset_path = "data/IrishFolkSong/session/" # dataset path
t_ec2vae = False
t_measure_vae = False

def main():
    process_irish(s_dir, dataset_path)
    if t_ec2vae: train_ec2vae()
    if t_measure_vae: train_measure_vae()

    process_data()

    train_sketch_vae()
    train_music_inpaintNet()
    train_sketchvae_inpaintRNN()
    train_sketchnet()

    evaluate_musicInpaintNet()
    evaluata_SketchVAE_InpaintRNN()
    evaluate_sketchNet()

    get_loss_and_accuracy()









if __name__ == "__main__":

    main()