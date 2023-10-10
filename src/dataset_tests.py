from src.datasets.ocr.test.ocr_datasets_unitary_testing import (log_dataset, try_esposalles, try_cocotext, try_funsd, try_washinton, try_hiertext, try_maps, try_iam,\
                                                                try_iii, try_mlt19, try_parzival, try_xfund, try_totaltext, try_textocr, try_svt, try_sroie, try_saint_gall)


### COCO ###
print('Whole COCO')
d1 = try_cocotext()

print("Coco Val")
d2 = try_cocotext(split='val')

print('English vs Non-English')
try_cocotext(langs=['english'])
try_cocotext(langs=['not english'])

print('Legible vs iLegible')
try_cocotext(legibility=['legible'])
try_cocotext(legibility=['illegible'])

### ESPOSALLES ###
print('Whole esposalles - words') # Train is 75% CV
d3 = try_esposalles(mode='words')

print('Whole esposalles - lines')
d4 = try_esposalles(mode='lines')

### FUNSD ###
print('whole funsd')
d5 = try_funsd()

print('funsd - test')
try_funsd(split = 'test')

### GW ###
print('Whole GW - words') 
d7 = try_washinton(split = 'train', mode = 'word')
print('Whole GW - lines')
try_washinton(split = 'train', mode = 'line')

print('Whole GW - val lines')
try_washinton(split = 'valid', mode = 'line')

print('Whole GW - val words')
try_washinton(split = 'valid', mode = 'word')

print('Whole GW - test lines')
try_washinton(split = 'test', mode = 'line')

print('Whole GW - test words')
try_washinton(split = 'test', mode = 'word')

## HierText ###
# print('Whole HierText')
# d8 = try_hiertext()

# print('Hier Text Val')
# try_hiertext(split='val')

# print("Hier Text HW")
# try_hiertext(handwritten=[True])
# print("HierText Printed")
# try_hiertext(handwritten=[False])
# print("HierText words")
# try_hiertext(mode='words')
# print("HierText lines")
# try_hiertext(mode = 'lines')

### Historical Maps ####
print('Historical Maps')
d9 = try_maps()

### IAM ###
print('IAM Dataset')
d10 = try_iam(partition = 'aachen', split = 'train')

print('iam validation')
try_iam(partition = 'aachen', split = 'val')

print('iam test')
try_iam(partition = 'aachen', split = 'test')


### IIIT5K
print('IIIT Dataset')
d11 = try_iii(split = 'train')
print('iii test')
try_iii(split = 'test')


### MLT19
print('MLT19 dataset')

langs = ['Latin', 'Arabic',  "Chinese", "Japanese", "Korean", "Bangla", "Hindi", "Symbols", "Mixed", "None"]

d12 = try_mlt19()

for ln in langs:
    print(f"MLT train {ln}")
    try_mlt19(language=[ln])

print('MLT19 val')
try_mlt19(split='val')


### PARZIVAL
print('Parzival dataset')
d13 = try_parzival(split = 'train', mode = 'word')

print('Parzival test word')
try_parzival(split = 'train', mode = 'word')

print('Parzival validation word')
try_parzival(split = 'valid', mode = 'word')

print('Parzival dataset line')
try_parzival(split = 'train', mode = 'line')

print('Parzival test line')
try_parzival(split = 'train', mode = 'line')

print('Parzival validation line')
try_parzival(split = 'valid', mode = 'line')

### Saint Gall
print('saint gall train')
d14 = try_saint_gall()

print('saint gall test')
try_saint_gall(split = 'test')

print('saint gall valid')
try_saint_gall(split = 'valid')

### SROIE
print('sroie train')
d15 = try_sroie(split='train')

print('sroie test')
try_sroie(split='test')

### SVT
print('svt train')
d16 = try_svt(split = 'train')
print('svt test')
try_svt(split = 'test')

## TextOCR
print('text ocr train')
d17 = try_textocr(split = 'train')

print('text ocr val')
try_textocr(split = 'val')

## TOTAL TEXT
print('total text')
d18 = try_totaltext(split = 'Train')

print('totaltext test')
try_totaltext(split = 'Test')

## xFund
print('xfund train')
langs = ['DE', 'ES', 'FR', 'IT', 'JA', 'PT', 'ZH']
d19 = try_xfund(split = 'train')

for lg in langs:
    print(f"xFund {lg}")
    try_xfund(lang=[lg])

print('xfund val')
try_xfund(split = 'val')
