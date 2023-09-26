import numpy as np
import librosa
import pydub
from pydub import AudioSegment
import dtaidistance

from music21 import *
from converter21 import tempo, M, razmer
mistakes_ind = []
secs = []
freqs_ref = []

def compare2 ( input_audio , M):
   #sound = AudioSegment.from_mp3(input_audio)
   #sound.export("MusicAssistant/Mozart/playing.wav", format="wav")

   #x, sr = librosa.load("MusicAssistant/Mozart/playing.wav")
   x,sr = librosa.load(input_audio)
   f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
   times = librosa.times_like(f0)
   otschet = 60/tempo
   print(otschet)
   neededtimes = []
   mids=[]
   a=0
   prevstep=0
   freqsrecognizedindexes = []
   freqsrecognized=[]
   #step_in_array = (x.shape[0]/sr)/times.shape[0]   #in secs
   freqs_low =[]
   freqs_high=[]
   #print(f0) 
   doubled=[] 
   for n in M.notes:
      freqs_ref.append(n.pitch.frequency)
      #print('f', n.pitch.frequency)
      #print('o', n.pitch.octave)
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

      #print('DURATION ', n.duration.type)
      if  n.duration.type == 'half' :   
          secs.append(otschet*2)
      if  n.duration.type == 'quarter':
          secs.append(otschet)
      if  n.duration.type == 'eight':
          secs.append(otschet/2)
   #print('Notes freqs: ', freqs_ref ) 

   #print('SECS' , secs)
   #print(times)
    
   for k in range (len(secs)):
       #a = a + secs[k]
       for j in range(prevstep, times.shape[0]-1):
           
           #print(a)
           #print(times[j])
           if ((times[j]<=a)and(times[j+1]>=a)and f0[j]>0 ):
               neededtimes.append(times[j])
               freqsrecognizedindexes.append(j)
               prevstep = j+1
       a = a + secs[k]       
                 
    
   print('Indexes ', freqsrecognizedindexes)
   print('Needed times ', neededtimes )
   print(' indexes for f0 ', freqsrecognizedindexes)
   print(len(neededtimes))
   print(len(freqsrecognizedindexes))

   
   for g in range (len(neededtimes)-1):
       mids.append((neededtimes[g+1]+neededtimes[g])/2)
       freqsrecognized.append(f0[(freqsrecognizedindexes[g+1] + freqsrecognizedindexes[g] )//2])
   #mids.append(((times[times.shape[0]-1])+neededtimes.pop())/2)   
   print('Mids', mids)
   print('AAAAAAAAAAAA', freqsrecognized)   
   print(freqs_ref)
   print(f0)


   '''for w in range(len(secs)-1):
       if ( ( abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_high[w] - freqs_ref[w])/3 ) or (abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_low[w] - freqs_ref[w])/3 )):
          mistakes_ind.append(w)
   print('Mistakes ',mistakes_ind ) '''
   
   '''for t in range(len(freqs_ref)):
      for x in range(22):
         doubled.append(freqs_ref[t])

   print(dtaidistance.dtw.distance_fast(np.array(freqs_ref), np.array(freqs_ref)))'''



   return mistakes_ind 


def compare3 ( input_audio , M):
   #sound = AudioSegment.from_mp3(input_audio)
   #sound.export("MusicAssistant/Mozart/playing.wav", format="wav")

   #x, sr = librosa.load("MusicAssistant/Mozart/playing.wav")
   x,sr = librosa.load(input_audio)
   f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
   times = librosa.times_like(f0)
   otschet = 60/tempo
   print(otschet)
   neededtimes = []
   mids=[]
   a=0
   r=0
   prevstep=0
   freqsrecognizedindexes = []
   freqsrecognized=[]
   #step_in_array = (x.shape[0]/sr)/times.shape[0]   #in secs
   freqs_low =[]
   freqs_high=[]
   #print(times) 

   for n in M.notes:
      freqs_ref.append(n.pitch.frequency)
      #print('f', n.pitch.frequency)
      #print('o', n.pitch.octave)
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

      #print('DURATION ', n.duration.type)
      if  n.duration.type == 'half' :   
          secs.append(otschet*2)
      if  n.duration.type == 'quarter':
          secs.append(otschet)
      if  n.duration.type == 'eight':
          secs.append(otschet/2)
   #print('Notes freqs: ', freqs_ref ) 

   #print('SECS' , secs)
   #print(times)
    
   for k in range (len(secs)):
       #a = a + secs[k]
       for j in range(prevstep, times.shape[0]-1):
           
           #print(a)
           #print(times[j])
           #if (((times[j]-0.5 <=a)and(times[j+1]+0.5 >=a)and f0[j]>0 )) and (abs(f0[j] - freqs_ref[k])< 5):
           if ((f0[j]>0)and (abs(f0[j] - freqs_ref[k])< 5)):
               neededtimes.append(times[j])
               freqsrecognizedindexes.append(j)
               prevstep = j+1
               break
       a = a + secs[k]       
                 
    
   print('Indexes ', freqsrecognizedindexes)
   print('Needed times ', neededtimes )
   print('Indexes for f0 ', freqsrecognizedindexes)
   print(len(neededtimes))
   print(len(freqsrecognizedindexes))

   
   for g in range (len(neededtimes)-1):
       mids.append((neededtimes[g+1]+neededtimes[g])/2)
       freqsrecognized.append(f0[(freqsrecognizedindexes[g+1] + freqsrecognizedindexes[g] )//2])


   for t in range(len(times)):
       if f0[len(times)-1-t]>0:
              mids.append ((neededtimes.pop() +times[len(times)-1-t] ) /2 ) 
              freqsrecognized.append(f0[(freqsrecognizedindexes.pop() + len(times)-1-t )//2])
              break
       


   #mids.append(((times[times.shape[0]-1])+neededtimes.pop())/2)   
   print('Mids', mids)
   print('AAAAAAAAAAAA', freqsrecognized)   
   print(freqs_ref)
   print(f0)
   '''for w in range(len(secs)-1):
       if ( ( abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_high[w] - freqs_ref[w])/3 ) or (abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_low[w] - freqs_ref[w])/3 )):
          mistakes_ind.append(w)
   print('Mistakes ',mistakes_ind ) '''
   return mistakes_ind 
    
    


def compare4 ( input_audio , M):
   #sound = AudioSegment.from_mp3(input_audio)
   #sound.export("MusicAssistant/Mozart/playing.wav", format="wav")

   #x, sr = librosa.load("MusicAssistant/Mozart/playing.wav")
   x,sr = librosa.load(input_audio)
   f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
   times = librosa.times_like(f0)
   otschet = 60/tempo
   print(otschet)
   neededtimes = []
   mids=[]
   a=0
   r=0
   prevstep=0
   freqsrecognizedindexes = []
   freqsrecognized=[]
   #step_in_array = (x.shape[0]/sr)/times.shape[0]   #in secs
   freqs_low =[]
   freqs_high=[]
   #print(times) 

   for n in M.notes:
      freqs_ref.append(n.pitch.frequency)
      #print('f', n.pitch.frequency)
      #print('o', n.pitch.octave)
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
   print('start',start)   
   a = times[start]  
   print(a)
   prevstep=start

   for k in range (len(secs)):
       a = a + secs[k]
       for j in range(prevstep, times.shape[0]-1):
           
           #print(a)
           #print(times[j])
           #if (((times[j]-0.5 <=a)and(times[j+1]+0.5 >=a)and f0[j]>0 )) and (abs(f0[j] - freqs_ref[k])< 5):
           if (((times[j] <=a)and(times[j+1] >=a)and f0[j]>0 )):
           #if ((f0[j]>0)and (abs(f0[j] - freqs_ref[k])< 5)):
               neededtimes.append(times[j])
               freqsrecognizedindexes.append(j)
               prevstep = j+1   #]]]]]]]
               break
       #a = a + secs[k]       
                  
   print('Indexes ', freqsrecognizedindexes)
   print('Needed times ', neededtimes )
   print('Indexes for f0 ', freqsrecognizedindexes)

   timestamps_from_corr= []
   detected_freqs  =[]
   p = 0
   Correlation = 0
   min = 100000
   #prevs = freqsrecognizedindexes[0]

   prevs = start

   print(prevs)
   for k in range (len(secs)-1):
      min=100000
      timestamps_from_corr.append(0)  
      detected_freqs.append(0)      
      Correlation = 0
      
      #for t in range (prevs - p*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//5 , prevs +  freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k] +  (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//5  ):
      for t in range (prevs - p*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//5 , prevs +  freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k] +  (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//5  ):
        for j in range (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]):
            
            if f0[t]>0 :
               #print('f0', f0[t] , freqs_ref[k])
               Correlation +=  f0[t]*freqs_ref[k]
               #print(k ,t ,j ,Correlation)

            if (abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) - freqs_ref[k]**2 )<min):
                min = abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) - freqs_ref[k]**2 )
                result_corr = Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])
                index_for_max_corr = t
                #print('detected' , t , np.sqrt(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])) , freqs_ref[k])
                timestamps_from_corr[k] =  t
                detected_freqs[k] =   np.sqrt(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]))
            #if (( result_corr < freqs_high[k]**2 )and (  result_corr> freqs_low[k]**2 )):
                #if (index_for_max_corr > 0 )and( abs(index_for_max_corr - freqsrecognizedindexes[k]) <=2 ):
                    #print("rythm is good ")
                #else:
                    #print("rythm is bad")  
      prevs = freqsrecognizedindexes[k]  + start
      p=1

      '''if (( result_corr < freqs_high[k]**2 )and (  result_corr> freqs_low[k]**2 )):
          if (index_for_max_corr > 0 )and( abs(index_for_max_corr - freqsrecognizedindexes[k]) <=2 ):
             print("rythm is good ")
          else:
             print("rythm is bad")   '''
      # maxcorr = 0

   print(freqs_ref) 
   print(detected_freqs)
   print(timestamps_from_corr)
   print(freqsrecognizedindexes )

   '''
   for g in range (len(neededtimes)-1):
       mids.append((neededtimes[g+1]+neededtimes[g])/2)
       freqsrecognized.append(f0[(freqsrecognizedindexes[g+1] + freqsrecognizedindexes[g] )//2])


   for t in range(len(times)):
       if f0[len(times)-1-t]>0:
              mids.append ((neededtimes.pop() +times[len(times)-1-t] ) /2 ) 
              freqsrecognized.append(f0[(freqsrecognizedindexes.pop() + len(times)-1-t )//2])
              break
       


   #mids.append(((times[times.shape[0]-1])+neededtimes.pop())/2)   
   print('Mids', mids)
   print('AAAAAAAAAAAA', freqsrecognized)   
   print(freqs_ref)
   print(f0)'''
   '''for w in range(len(secs)-1):
       if ( ( abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_high[w] - freqs_ref[w])/3 ) or (abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_low[w] - freqs_ref[w])/3 )):
          mistakes_ind.append(w)
   print('Mistakes ',mistakes_ind ) '''



   return mistakes_ind 
    
    
def compare5 ( input_audio , M):
   #sound = AudioSegment.from_mp3(input_audio)
   #sound.export("MusicAssistant/Mozart/playing.wav", format="wav")
   #x, sr = librosa.load("MusicAssistant/Mozart/playing.wav")


   x,sr = librosa.load(input_audio)
   f0, voiced_flag, voiced_probs = librosa.pyin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'))
   times = librosa.times_like(f0)
   otschet = 60/tempo
   #print(otschet)
   neededtimes = []
   mids=[]
   a=0
   r=0
   prevstep=0
   freqsrecognizedindexes = []
   freqsrecognized=[]
   #step_in_array = (x.shape[0]/sr)/times.shape[0]   #in secs
   freqs_low =[]
   freqs_high=[]
   #print(times) 

   for n in M.notes:
      freqs_ref.append(n.pitch.frequency)
      #print('f', n.pitch.frequency)
      #print('o', n.pitch.octave)
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

      #print('DURATION ', n.duration.type)
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

   print('start',start)   
   print('break', ending)

   a = times[start]  
   prevstep=start
   print(a)

   '''prevstep = 0
   start = 0
   a=0'''
  

   for k in range (len(secs)):
       #a = a + secs[k]
       print(a)
       for j in range(prevstep, times.shape[0]-1):
           
           #if (((times[j]-0.5 <=a)and(times[j+1]+0.5 >=a)and f0[j]>0 )) and (abs(f0[j] - freqs_ref[k])< 5):
           if (((times[j] <=a)and(times[j+1] >=a)and f0[j]>0 )):
               print(j,times[j])
           #if ((f0[j]>0)and (abs(f0[j] - freqs_ref[k])< 5)):
               neededtimes.append(times[j])
               freqsrecognizedindexes.append(j)
               prevstep = j+1   #]]]]]]]
               break
       a = a + secs[k]       
                  
   print('Indexes ', freqsrecognizedindexes)
   print('Needed times ', neededtimes )
   print('Indexes for f0 ', freqsrecognizedindexes)
   print(len(secs))
   timestamps_from_corr= []
   detected_freqs  =[]
   p = 0
   Correlation = 0
   min = 100000
   prevs = freqsrecognizedindexes[0]

   prevs = start
   r=1
   l = 0
   ii=0
   freqsrecognizedindexes.append(int(freqsrecognizedindexes[-1] + (secs[-1])//(times[1]-times[0]))-1) #new

   for k in range (len(secs)):
      min=10000000
      timestamps_from_corr.append(0)  
      detected_freqs.append(0)    


      if(k>0):   
         prevs = timestamps_from_corr[k-1] + freqsrecognizedindexes[k+1] -   freqsrecognizedindexes[k]
         print('prevs' , prevs)
      if(k==len(secs)-1):
         r=0
         l=1   
         print('AAAAAAA', prevs , freqsrecognizedindexes[k+1] -   freqsrecognizedindexes[k] , ending , ending - (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])-prevs-1 )
      #print('jjjjjj',freqsrecognizedindexes[k] , prevs)
      #for t in range (prevs - p*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//5 , prevs +  freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k] +  (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//5  ):
      for t in range (prevs - p*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) //2, prevs +  r*(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])//2 + l*(ending - (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k])-prevs+1)):
        
        for j in range (freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]):

            if f0[t+j]>0 :

               Correlation +=  f0[t+j]*freqs_ref[k]
            else:
               Correlation +=  0   
               ii=ii+1

            if ( (j==freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-1)and(abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]) - freqs_ref[k]**2 )<min)):
                min = abs(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-ii) - freqs_ref[k]**2 )
                '''if(k==2):
                   
                  print(k,prevs,t,j,t+j,min)'''
                result_corr = Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-ii)
                index_for_max_corr = t

                timestamps_from_corr[k] =  t
                detected_freqs[k] =   np.sqrt(Correlation/(freqsrecognizedindexes[k+1] - freqsrecognizedindexes[k]-ii))
            #if (( result_corr < freqs_high[k]**2 )and (  result_corr> freqs_low[k]**2 )):
                #if (index_for_max_corr > 0 )and( abs(index_for_max_corr - freqsrecognizedindexes[k]) <=2 ):
                    #print("rythm is good ")
                #else:
                    #print("rythm is bad")  
        Correlation=0            
        ii=0
      #prevs = freqsrecognizedindexes[k+1] 
      p=1

      '''if (( result_corr < freqs_high[k]**2 )and (  result_corr> freqs_low[k]**2 )):
          if (index_for_max_corr > 0 )and( abs(index_for_max_corr - freqsrecognizedindexes[k]) <=2 ):
             print("rythm is good ")
          else:
             print("rythm is bad")   '''
      # maxcorr = 0
   # freqsrecognizedindexes.pop()

   print(freqs_ref) 
   print(detected_freqs)

   print(timestamps_from_corr)
   print(freqsrecognizedindexes[:-1])
   '''
   for g in range (len(neededtimes)-1):
       mids.append((neededtimes[g+1]+neededtimes[g])/2)
       freqsrecognized.append(f0[(freqsrecognizedindexes[g+1] + freqsrecognizedindexes[g] )//2])
   '''
   '''
   for t in range(len(times)):
       if f0[len(times)-1-t]>0:
              mids.append ((neededtimes.pop() +times[len(times)-1-t] ) /2 ) 
              freqsrecognized.append(f0[(freqsrecognizedindexes.pop() + len(times)-1-t )//2])
              break'''
       


   #mids.append(((times[times.shape[0]-1])+neededtimes.pop())/2)   

   
   '''for w in range(len(secs)-1):
       if ( ( abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_high[w] - freqs_ref[w])/3 ) or (abs(freqsrecognized[w] - freqs_ref[w]) >= abs(freqs_low[w] - freqs_ref[w])/3 )):
          mistakes_ind.append(w)
   print('Mistakes ',mistakes_ind ) '''



   return mistakes_ind 