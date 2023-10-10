# Common OCR Dataloaders

I created a collection of dataloaders for _recognition_ task. Although detection should be easily implemented, this dataloaders only incorporate recognition (it's free, don't complain).
For internal CVC distribution only amolina@cvc.uab.cat has the original .zip file containing all the data weakly curated.

Make sure to read Terms and Conditions for each dataset before using it.

## Dataset Wiki:

  * Esposalles: Historical catalan handwritten recognition with word and line level annotations. 4 75% cross validation sets are provided as well. (source)[http://dag.cvc.uab.es/the-esposalles-database/]
    - Train Words 75% of 23588 (4-Fold Cross validation)
    - Train Lines: 75% of 2288
  * COCOText: English and non-English COCO Images annotated with OCR and manually corrected. Scene text recognition and a few handwritted images. (source)[https://bgshih.github.io/cocotext/]:
    - Train words: 163476 (97557 ilegible)
        1. English: 157579
        2. Not English: 5897
    - Val Words: 37650
      
  * FUNSD: English documents for form text (printed and some handwritten) text recognition. (source)[https://bgshih.github.io/cocotext/]
     - Train entities: 22512
     - Test entities: 8973
  * George Washington: English historical handwritten recognition with word and line level annotations (aachen splits provided). (source)[https://fki.tic.heia-fr.ch/databases/washington-database]
    - Train Words: 2433
    - Train lines: 325
    - Val lines: 168
    - val words: 1293
    - test words: 1168
    - test lines: 163
  * HierText: Handwritten and printed scene text recognition with hierarchical annotation (paragraph, line, words). (source)[https://github.com/google-research-datasets/hiertext]
     - Train words: 925545
         1. Handwritten: 63100
         2. Other: 862445
      - Validation words: 191433
      - Train lines: 435138
   * Historical Maps: English printed recognition in historical maps. (source)[https://weinman.cs.grinnell.edu/research/maps.shtml#data]
      - Train words: 75% of 25478 (4-fold cross validation)

   * IAM: Good old IAM. Handwritten author-wise text recognition at line and word level. (source)[https://fki.tic.heia-fr.ch/databases/iam-handwriting-database]
Using aachen partitions as found https://github.com/jpuigcerver/Laia/tree/master/egs/iam
     - Train (aachen split) words: 14331
     - Val (aachen split) words: 413
     - Test (aachen split) words: 1422
 
  * IIIT5K: Scene text recognition in english. (source)[https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset].
      - Train words 3000
      - Test words 2000
 * MLT19 Dataset: Multilingual scene text. (source)[https://rrc.cvc.uab.es/?ch=15]
     - Train words: 82662
        1. latin: 51296
        2. arabic: 4013
        3. chinese: 2545
        4. japanese: 4920
        5. korean: 5447
        6. bangla: 3169
        7. hindi: 3033
        8. symbols: 1295
        9. mixed: 179
        10. None: 6765
      - Val words: 29336 (from 4-fold cv, test evaluation can be found in rrc site)

* Parzival: Historical german binarized documents with word and line level annotation. (source)[https://fki.tic.heia-fr.ch/databases/parzival-database]
        - Train words: 5872
        - Test words: TBA
        - Validation words: 2936
        - Train lines: 2237
        - Test lines: TBA
        - Validation lines: 912

* Saint Gall: Historical handwritten latin documents at line level (there's word-level annotatins but they don't work for me :( ). (source)[https://fki.tic.heia-fr.ch/databases/saint-gall-database]
         - Train lines: 707
         - Test lines: 468
         - Validation lines: 235

* SROIE: Superset (i think) of FUNSD for receipts. (source)[https://rrc.cvc.uab.es/?ch=13]
           - Train entities: 34930
           - Test entities: 19386

* SVT: Street view images for scene text recognition. A link doesn't seem to work, idk why but i have the images. (source)[https://tc11.cvc.uab.es/datasets/SVT_1]
            - Train words: 647
            - Test words: 257

* TextOCR: OCR-Extracted annotations for VQA. (source)[https://textvqa.org/textocr/dataset/]
             - train words: 714770
             - val words: 107802

* TOTAL TEXT: Totally cool dataset for curved text recognition (interesting for domain shift reasons). (source)[https://github.com/cs-chan/Total-Text-Dataset/tree/master/Groundtruth]
             - Train words: 10589
             - Test words: 2547

* XFund: multilingual FUNSD with 199 pages per language. (source)[https://github.com/doc-analysis/XFUND/releases/tag/v1.0].
             - Training entities: 71999
                1. DE: 8632
                2. ES: 11449
                3. FR: 8816
                4. IT: 12215
                5. JA: 9005
                6. PT: 11654
                7. ZH: 10228
           - Validation entities: 25013
  
    
Note: Some measurements may be messed up, give me time I spent +3 whole days on that and i get easily overwhelmed :(
