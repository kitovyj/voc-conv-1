import os

#for i in range(5760 - 250):
for i in range(250):

    fn = "test" + str(i).zfill(9) + "r" + ".png"
    if not os.path.isfile(fn):
        print('no file ', i)
    #else:
    #    print(i)
