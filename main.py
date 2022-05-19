from copyreg import pickle
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
import numpy as npy
from loader.dataloader import MIDI_Render
from MeasureVAE.measure_vae import MeasureVAE
from SketchNet.sketchnet import SketchNet
from SketchVAE.sketchvae import SketchVAE
from InpaintRNN.inpaintrnn import  InpaintingNet





s_dir = "" # folder address
dataset_path = "data/IrishFolkSong/session/" # dataset path
t_ec2vae = False
t_measure_vae = False

def main():
    measure_vae = MeasureVAE()
    print(measure_vae.parameters())

    sketchnet = SketchNet()
    print(sketchnet.parameters())

    sketch_vae = SketchVae()
    print(sketch_vae.parameters())
    
    inpaint_rnn = InpaintingNet()
    print(inpaint_rnn.parameters())


    # process_irish(s_dir, dataset_path)
    # if t_ec2vae: train_ec2vae()
    # if t_measure_vae: train_measure_vae()

    # process_data()

    # train_sketch_vae()
    # train_music_inpaintNet()
    # train_sketchvae_inpaintRNN()
    # train_sketchnet()

    # evaluate_musicInpaintNet()
    # evaluata_SketchVAE_InpaintRNN()
    # output = evaluate_sketchNet()
    # output = npy.load('res-exp-sketchnet_control_mixture_2.npy', allow_pickle= True)
    # mid =  MIDI_Render('Irish')
    # output_midi = mid.data2midi(output)
    # print(output)

    # get_loss_and_accuracy()









if __name__ == "__main__":

    main()