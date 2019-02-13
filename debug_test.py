from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ImgData as ID
import Train as T
T.train('D:\iMI\im2latex_train.lst', 'D:\iMI\im2latex_formulas_utf.lst', 'D:\iMI\data', 32)
'''shape = (320, 80)
trainset = ID.SmallImgDataSet(24, 'D:\iMI\im2latex_train.lst', 'D:\iMI\im2latex_formulas_utf.lst', 'D:\iMI\data', transform=transforms.Compose([
                                      ID.UniChannel(),
                                      ID.Rescale(shape),
                                      ID.ToTensor()
                                  ]))
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
for i, data in enumerate(trainloader, 0):
    print(data['latex'][0])
    break'''