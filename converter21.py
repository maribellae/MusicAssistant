#import music21
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
import music21
from music21 import *
from music21 import stream
from music21 import note
from music21.note import Note
from glob import glob
M = stream.Stream()
tempo = 120
razmer = 4
def converter1(input_path):
    my_path = sorted(glob(f'{input_path}/output/code/*'))
    i_path = my_path[0]
    notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    signs = ['#' , '&']
    durs = ['2', '4', '8' ]
    mydurs = [ 'half', 'quarter', 'eight']
    octavs = ['1' , '2']
    file = open(i_path, 'r')

    #M = stream.Measure()
    M.append(clef.TrebleClef())
    M.append(music21.tempo.MetronomeMark(number=tempo))
    prevsign = ''
    mynote=''
    i=0
    detector=0
    octava = ''
    while 1:  
        # read by character
        char = file.read(1) 
        # print(char)        
        if not char :
            break
        else: 
            if char == '/':
                detector = 1
            #print ('A ')
            if char in signs:
                if char == '&':
                    prevsign = '-'
                else:
                    prevsign = '&'    
            if char in notes:
               mynote = char.upper()
            if char in octavs and detector ==0 :
                if char =='1':
                    octava = '4'
                else:
                    octava = '5'       
            if char in durs and detector ==1 :
               while i < 3:
                  if char == durs[i]:
                      print(mynote + prevsign + octava +mydurs[i])
                      M.append(note.Note(mynote + prevsign + octava, type=mydurs[i]))
                  i=i+1    
               i=0
               prevsign = ''
               mynote=''
               detector = 0
        #if char ==']':
        #    break        
    #M.show()  
    # 
    '''mf = midi.translate.streamToMidiFile(M)
    mf.open('MusicAssistant/Mozart/result.mid', 'wb')
    mf.write()
    mf.close() 
    #M.write('midi' , 'Mozart/result.mid' )'''
    file.close()

    #return M


def transpozition(path , neededkey, type):
    inputstream= converter1(path)
    #print(inputstream)
    k = inputstream.analyze('key')

    k = k.getScale(type)

    i = interval.Interval(k.tonic, pitch.Pitch(neededkey))

    sNew = inputstream.transpose(i)
    k = sNew.analyze('key')

    k=k.getScale(type)    

    sNew.key = k

    '''mf = midi.translate.streamToMidiFile(sNew)
    mf.open('MusicAssistant/Mozart/transpozes_result.mid', 'wb')
    mf.write()
    mf.close() '''

    return sNew    
