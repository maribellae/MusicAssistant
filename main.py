
from Mozart.src.mozartworker import *
from newscanner import *
from converter21 import *
from comparison_tool import *
#M = stream.Stream()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("inputfolder", help="Input File")
    parser.add_argument("option" , help = "Function to do")
    parser.add_argument("Key1" , help = 'c/d/e/f/g/a/b')
    parser.add_argument("Key2" , help = 'major/minor')    
    args = parser.parse_args()
    option = args.option

    if  option =='scan':
       #scan(args.inputfolder)
       #scan2()
       mozartgetnotes(args.inputfolder)
    elif option =='convert':
       transpozition(args.inputfolder, args.Key1, args.Key2)
    elif option == 'play':
       converter1(args.inputfolder)
       indexes1, indexes2 = compare(args.inputfolder,M)
       mozartworkmistakes(args.inputfolder, indexes1, indexes2  )



    #converter('Mozart/output/txt/04.txt')
    #print( converter('Mozart/output/txt/04.txt'))
    #transpozition('Mozart/output/txt/04.txt','g','major')

    #print(indexes1 , indexes2)

    #print(indexes)
    #scan2()
    #mozartwork(args.inputfolder, args.outputfolder)

