echo 'generate sample from postive image'

for file in ./positive_images/* 
do
	sub=${file##*/}
	num=${sub%.*}
	opencv_createsamples -img $file -vec ./vectors/positive/$num.vec -maxxangle 0.2 -maxyangle 0.2 -maxzangle 0.2 -w 32 -h 32
done

# echo 'generate sample from negative image'

# for file in ./negative_images/* 
# do
# 	sub=${file##*/}
# 	num=${sub%.*}
# 	opencv_createsamples -img $file -vec ./vectors/negative/$num.vec -randinv -maxxangle 2.0 -maxyangle 2.0 -maxzangle 2.0
# done

echo 'convert positive vector to images'

for file in ./vectors/positive/* 
do
	python showvec.py -i $file -f positive 
done

# echo 'convert negative vector to images'

# for file in ./vectors/negative/* 
# do
# 	python showvec.py -i $file -f negative
# done
