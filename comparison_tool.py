import numpy as np
import librosa
import pydub
from pydub import AudioSegment
import dtaidistance
import math
from music21 import *
from glob import glob
from converter21 import tempo,  razmer  #MMMMMMMMMMMMMMMM
mistakes_pitch_ind_and_feedback = []
mistakes_duration_ind_and_feedback = []
secs = []
freqs_ref = []
import os
def compare ( input_path , M):
   #sound = AudioSegment.from_mp3(input_audio)
   #sound.export("MusicAssistant/Mozart/playing.wav", format="wav")
   #x, sr = librosa.load("MusicAssistant/Mozart/playing.wav")
   audio_path = sorted(glob(f'{input_path}/input/audio/*'))

   i_path = audio_path[0]
   i_path = i_path.replace(os.sep, '/')
   
   x,sr = librosa.load(i_path)
   f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
   
   times = librosa.times_like(f0)
   otschet = 60/tempo
   neededtimes = []
   mids=[]
   a=0
   r=0
   prevstep=0
   freqsrecognizedindexes = []
   freqsrecognized=[]
   freqs_low =[]
   freqs_high=[]

   
   for n in M.notes:
      freqs_ref.append(n.pitch.frequency)

      if(n.pitch.name == 'B' ):
        freqs_high.append(  pitch.Pitch('C'+ str(n.pitch.octave +1 )).frequency  )
        freqs_low.append(  pitch.Pitch('A#'+ str(n.pitch.octave ) ).frequency  )
      elif(n.pitch.name == 'C' ):
        freqs_high.append(  pitch.Pitch('C#'+str(n.pitch.octave )).frequency  )
        freqs_low.append(  pitch.Pitch('B'+ str(n.pitch.octave - 1 )).frequency  )   
      elif(n.pitch.name == 'D' ):
        freqs_high.append(  pitch.Pitch('D#'+ str(n.pitch.octave  )).frequency  )
        freqs_low.append(  pitch.Pitch('C#'+ str(n.pitch.octave) ).frequency  )   
      elif(n.pitch.name == 'G' ):
        freqs_high.append(  pitch.Pitch('G#'+ str(n.pitch.octave ) ).frequency  )
        freqs_low.append(  pitch.Pitch('F#'+ str(n.pitch.octave ) ).frequency  )                 
      elif(n.pitch.name == 'F' ):
        freqs_high.append(  pitch.Pitch('F#'+ str(n.pitch.octave  )).frequency  )
        freqs_low.append(  pitch.Pitch('E'+ str(n.pitch.octave ) ).frequency  )             
      elif(n.pitch.name == 'A' ):
        freqs_high.append(  pitch.Pitch('A#'+ str(n.pitch.octave  )).frequency  )
        freqs_low.append(  pitch.Pitch('G#'+ str(n.pitch.octave ) ).frequency  )   
      elif(n.pitch.name == 'E' ):
        freqs_high.append(  pitch.Pitch('F'+ str(n.pitch.octave ) ).frequency  )
        freqs_low.append(  pitch.Pitch('D#'+str(n.pitch.octave )).frequency  ) 


      if  n.duration.type == 'half' :   
          secs.append(otschet*2)
      if  n.duration.type == 'quarter':
          secs.append(otschet)
      if  n.duration.type == 'eight':
          secs.append(otschet/2)

   for y in range(times.shape[0]):
      if f0[y]>0:
         start = y
         break

   for j in range(times.shape[0]):
      if f0[times.shape[0]-1-j]>0:
         ending = times.shape[0]-1-j
         break

   a = times[start]  
   prevstep=start

  
   for k in range (len(secs)):

       for j in range(prevstep, times.shape[0]-1):

           if (((times[j] <=a)and(times[j+1] >=a)and f0[j]>0 )):

               neededtimes.append(times[j])
               freqsrecognizedindexes.append(j)
               prevstep = j+1   
               break
       a = a + secs[k]       
                  
 
   timestamps_from_corr= []
   detected_freqs  =[]
   p = 0
   Correlation = 0
   #prevs = freqsrecognizedindexes[0]
   prevs = start
   r=1
   l = 0
   ii=0
   prev_pitch=0
   freqsrecognizedindexes.append(int(freqsrecognizedindexes[-1] + (secs[-1])//(times[1]-times[0]))-1) #new

   for k in range (len(secs)):
      min=10000000
      timestamps_from_corr.append(0)  
      detected_freqs.append(0)    

      if(k>0):   
         prevs = timestamps_from_corr[k-1] + freqsrecognizedindexes[k+1] -   freqsrecognizedindexes[k]

      if(k==len(secs)-1):
         r=0
         l=1
      indikator = 0   
      for t in range (prevs - p*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) //2, prevs +  r*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) + l*(ending - (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])-prevs+2)):       
        for j in range (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]):

            if ((f0[t+j]>0 )and( abs(f0[t+j]-freqs_ref[k] ) <=  abs(freqs_high[k] - freqs_low[k])) ):
                indikator += 1    
                
                Correlation +=  f0[t+j]*freqs_ref[k]


                if ( (j==freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-1)and(abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k] ) - freqs_ref[k]**2 )<min)):
                    min = abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) - freqs_ref[k]**2 )
                    timestamps_from_corr[k] =  t
                    detected_freqs[k] =   np.sqrt(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]))

                #prev_pitch =    f0[t+j]     
                
            elif (math.isnan(f0[t+j])) :

                indikator+=0
                print(k)
                ii=ii+1
                    
                    
                if ( (j==freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-1)and(abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k] -ii) - freqs_ref[k]**2 )<min)):
                    min = abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-ii) - freqs_ref[k]**2 )
                    timestamps_from_corr[k] =  t
                    detected_freqs[k] =   np.sqrt(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-ii))


           

        Correlation=0            
        ii=0

      p=1

   freqsrecognizedindexes.pop()
   '''print('REF FREQS','\n', freqs_ref) 
   print('DETECTED FREQS', '\n',detected_freqs)

   print('DETECTED BEGINING ','\n',timestamps_from_corr)
   print('NEEDED BEGINING','\n', freqsrecognizedindexes)'''
   for w in range(len(secs)):
      if ( ( abs(detected_freqs[w] - freqs_ref[w]) >= abs(freqs_high[w] - freqs_ref[w])/3 ) or (abs(detected_freqs[w] - freqs_ref[w]) >= abs(freqs_low[w] - freqs_ref[w])/3 )):
          mistakes_pitch_ind_and_feedback.append(w )
          mistakes_pitch_ind_and_feedback.append( bool(detected_freqs[w] - freqs_ref[w] > 0 ) )
      if ((abs(timestamps_from_corr[w] - freqsrecognizedindexes[w]) >= int(otschet//(times[1]-times[0])))):
          mistakes_duration_ind_and_feedback.append(w)  
          mistakes_duration_ind_and_feedback.append(bool(timestamps_from_corr[w] - freqsrecognizedindexes[w] > 0 ) )

   print(mistakes_pitch_ind_and_feedback, mistakes_duration_ind_and_feedback )

   return mistakes_pitch_ind_and_feedback, mistakes_duration_ind_and_feedback 
