We use Snli data from https://nlp.stanford.edu/projects/snli/
And change the format to 
{"origin": " \"Cafe Express\" sign covered in graffiti.", "entailment": "A sign covered in graffiti.", "contradiction": "the sign is brand new and clean."}
{"origin": "# 6 tries her best to help her team to victory.", "entailment": "A female person is playing a team sport.", "contradiction": "A sports arena is empty."}
{"origin": "# 8 for Iowa State stiff arms a Texas AM player attempting to tackle him.", "entailment": "A player for Texas AM gets stiff armed.", "contradiction": "The player dropped the baseball."}
{"origin": "' The Herald' being sold and advertised at a mini-mart.", "entailment": "'The Herald' at a mini-mart.", "contradiction": "'The Daily News' at a local mall."}
{"origin": "1 guys running with a white ball football in sand while another chases him.", "entailment": "The boys are outside.", "contradiction": "The boys are in grass."}
{"origin": "1 little boy wearing a pirate costume following closely behind a little girl wearing a blue dress carrying a orange pumpkin bucket and walking down the sidewalk.", "entailment": "Boy in costume followed by a girl in costume.", "contradiction": "A boy in a clown costume followed by a girl."}
{"origin": "1 man is shaving another man's head in a room with a dirt floor.", "entailment": "A man is getting his head shaved.", "contradiction": "Two men watch a movie together."}
{"origin": "1 man riding a bike through the country.", "entailment": "A human is riding a bike.", "contradiction": "One man is riding a bike in the city."}
{"origin": "1 man singing and 1 man playing a saxophone in a concert.", "entailment": "The 2 men are entertaining some people.", "contradiction": "The 2 men are standing in the crowd holding guns."}
{"origin": "1 man standing and several people sitting down waiting on a subway train.", "entailment": "People waiting.", "contradiction": "A group of dogs chase a cat."}
{"origin": "1 man with rifle in hand and other instructing at a shooting range", "entailment": "Instructers helping at a shooting range.", "contradiction": "Two men with rifles meet each other for coffee."}


Test dev are from https://github.com/brmson/dataset-sts


We trained our model and saved it as .bin

First, download bert-based-unbased in https://huggingface.co/bert-base-uncased/tree/main and rename it as pretrained_model

Then replaced hugging face .bin file with our trained model .bin, rerun preprocess.py, analyze.py, visualize.py
   using the same procedure. **But remember to change model path in preprocess.py**

The result shown that contrastive learning model could alleviate the anisotropy problem.