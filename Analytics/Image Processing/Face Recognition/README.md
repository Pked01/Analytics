
# Facial Recognition
##  Applications:
### --> Surveillance system: Recognizing person(criminal/fugitive etc.) accross a video stream or CCTV.
### -->  Identification/ Login or attendence system.
<img src="image_1.png" alt="model in action">


## Model Theory
### 1. CNN based model is used to detect Face(Results in very less number of false negatives
### 2. Used Face-landmarks as feature to recognize faces(classification)
<img src="image_2.png" alt="Landmarks on Face">

### 3. Data preprocessing: 
#### Data images are filtered(containing multiple faces)
#### --> Class imbalance is removed using resampling and SMOTE
#### --> Partial Fit is used to incorporate incremental learning, so that data can repeatedly trained on old model without retraining as a whole

### libraries required:
* Sklearn
* Pandas,numpy
* Pandas_ml
* OpenCV,face_recognition(prebuild functions for dlib),dlib



```python
from facial_recognition import *
%matplotlib inline
```

#### Saving Directory


```python
data_path='data_folder/'
```

#### Collecting Data
* Data is collected via google images utility is developed which can download directly from google images
*  Assumption(n=100) images from google are from same character(query) search for

#### Testing
* For testing purpose we use cast list of different TV series and build model on top of that
* For testing purpose TVDB or TMDB are used for getting cast list


##### TVDB


```python
series_name='friends'
cast_list=get_cast_name_tvdb(series_name)
print(cast_list,len(cast_list))
```

    ([u'Lisa Kudrow', u'Matt LeBlanc', u'Matthew Perry', u'Courteney Cox', u'David Schwimmer', u'Jennifer Aniston'], 6)


##### TMDB


```python
series_name='friends'
cast_list=get_cast_name_tmdb(series_name)
print(cast_list,len(cast_list))
```

    ([u'Courteney Cox', u'Matt LeBlanc', u'Jennifer Aniston', u'David Schwimmer', u'Lisa Kudrow', u'Matthew Perry'], 6)


#### Downloading Data
* once cast list is generated same can be used for downloading image, using <i> downloaded_images</i>
* keywords : Extra text which is appended to each character in cast_list to filter down search more, ex Matt LeBlanc friends where friends is keyword


```python
downloaded_images(data_path=data_path+series_name+'/',cast_list=cast_list,keywords=[series_name]*len(cast_list))
```

#### Prepare Data
-returns [X,y] write label file in models folder under data_path  

* prepare data by passing reading path of images, minimum number of images(optional), maximum number of images(optional) and dump path
* only those folder will be read which have minimum number of images(l_threshold), only those folder will be read which have minimum number of images(l_threshold default 20), 
* if number of images is greater than r_threshold that folder is ignored(default is None)



```python
[X,y]=prepare_data(data_path=data_path+series_name+'/',l_threshold=10,dump_file_path=data_path+series_name+'/')
#[X,y]=pickle.load(open(data_path+series_name+'/'+'_encoded_file.pickle','rb'))
```

    dumping output
    returning prepare data


#### Training model
* Incremental training is done using partial fit utility of sk-learn, thus trained model can be pass again as bas_more for new data 
* Wrote a custom function to train model, which take care of unbalanced classes using SMOTE and resampling
* internally it will train multi based model until threshold_accuracy is reached, in each iteration training data is reshuffled. 
* number of retraining can be control by param n_retrain



```python
clf_sgd=SGDClassifier(loss='log',n_jobs=7,\
                      shuffle=True,class_weight=None,warm_start=False\
                      ,n_iter = np.ceil(10**6 / 600),average=True)
clf_sgd=train_model(clf_sgd,X,y,minm_image_process=100,threshold_accuracy=0.82,classes=list(range(1,10)),n_retrain=10)
```

    entering training module
    StratifiedShuffleSplit(n_splits=10, random_state=0, test_size=0.2,
                train_size=None)
    inside preprocessing function
    returning from preprocess data
    classes must be passed on the first call to partial_fit.


    /usr/local/lib/python2.7/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)
    /usr/local/lib/python2.7/site-packages/sklearn/linear_model/stochastic_gradient.py:117: DeprecationWarning: n_iter parameter is deprecated in 0.19 and will be removed in 0.21. Use max_iter and tol instead.
      DeprecationWarning)


    ('accuracy in iteration ', 1, ' is =', 0.94936708860759489)
    returning from train module



```python
pickle.dump(clf_sgd,open(data_path+series_name+'/'+'models/sgd_classifier.pickle','wb'),protocol=2)
```


```python
clf_sgd=pickle.load(open(data_path+series_name+'/'+'models/sgd_classifier.pickle','rb'))
```

#### Testing model
* Testing can be done on any video(frames),
* for downloading video we can use youtube-dl

*We can use -F flag to check the format code and then -f flag to download that video


```python
!youtube-dl -F https://www.youtube.com/watch?v=7qdwFQgMyVs
```

    [youtube] 7qdwFQgMyVs: Downloading webpage
    [youtube] 7qdwFQgMyVs: Downloading video info webpage
    [youtube] 7qdwFQgMyVs: Extracting video information
    [info] Available formats for 7qdwFQgMyVs:
    format code  extension  resolution note
    249          webm       audio only DASH audio   52k , opus @ 50k, 1.62MiB
    250          webm       audio only DASH audio   69k , opus @ 70k, 2.05MiB
    171          webm       audio only DASH audio  121k , vorbis@128k, 3.54MiB
    140          m4a        audio only DASH audio  127k , m4a_dash container, mp4a.40.2@128k, 4.36MiB
    251          webm       audio only DASH audio  134k , opus @160k, 3.86MiB
    278          webm       256x144    144p  110k , webm container, vp9, 30fps, video only, 3.38MiB
    160          mp4        256x144    144p  111k , avc1.4d400c, 30fps, video only, 1.93MiB
    133          mp4        426x240    240p  201k , avc1.4d4015, 30fps, video only, 3.40MiB
    242          webm       426x240    240p  242k , vp9, 30fps, video only, 6.58MiB
    243          webm       640x360    360p  452k , vp9, 30fps, video only, 12.15MiB
    134          mp4        640x360    360p  476k , avc1.4d401e, 30fps, video only, 8.33MiB
    244          webm       854x480    480p  820k , vp9, 30fps, video only, 20.53MiB
    135          mp4        854x480    480p  947k , avc1.4d401f, 30fps, video only, 16.93MiB
    247          webm       1280x720   720p 1644k , vp9, 30fps, video only, 42.84MiB
    136          mp4        1280x720   720p 1785k , avc1.4d401f, 30fps, video only, 34.22MiB
    248          webm       1920x1080  1080p 2935k , vp9, 30fps, video only, 85.13MiB
    137          mp4        1920x1080  1080p 3420k , avc1.640028, 30fps, video only, 72.08MiB
    17           3gp        176x144    small , mp4v.20.3, mp4a.40.2@ 24k
    36           3gp        320x180    small , mp4v.20.3, mp4a.40.2
    43           webm       640x360    medium , vp8.0, vorbis@128k
    18           mp4        640x360    medium , avc1.42001E, mp4a.40.2@ 96k
    22           mp4        1280x720   hd720 , avc1.64001F, mp4a.40.2@192k (best)



```python
!youtube-dl -f 137 https://www.youtube.com/watch?v=7qdwFQgMyVs
```

    [youtube] 7qdwFQgMyVs: Downloading webpage
    [youtube] 7qdwFQgMyVs: Downloading video info webpage
    [youtube] 7qdwFQgMyVs: Extracting video information
    [download] Destination: Friends - HD - The Videotape-7qdwFQgMyVs.mp4
    [K[download] 100% of 72.08MiB in 01:18.75KiB/s ETA 00:0091


##### testing on a video 
*get_pred_on_frame returns prediction on frame*


```python
get_video_processed('Friends - HD - The Videotape-7qdwFQgMyVs.mp4',data_path=data_path+series_name+'/'\
                    ,model=clf_sgd,skip_frames=10)
```

#### processed output
<img src="image_3.png" alt="Output of Model">
- Note: any output with less than .8 probablilty is not correct

#### output at 1FPS


```python
Disp.IFrame(data_path+series_name+'/'+'output.mp4',width=640,height=480)
```





        <iframe
            width="640"
            height="480"
            src="data_folder/friends/output.mp4"
            frameborder="0"
            allowfullscreen
        ></iframe>
        


