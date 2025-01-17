Here are some examples of questions you can ask the meal planner chatbot.

* Can you please suggest 2 chicken recipes and 3 vegetarian recipes
* I'd like to try some curry recipes
* I'm on a diet. Can you suggest some healthy meals.
* Can you come up with an original recipe that contains M&Ms
    * It should come up with a recipe not in the master list. 
    * You might notice it "hallucinates" the URL like https://www.example.com. There should actually be no URL for the recipe since it's an original recipe. You can try tweaking the system prompt in the code to see if you can fix the URL, so it returns something like 'N/A'.
* Give me 5 recipes with red meat
* I have a lot of carrots sitting around. Can you suggest some recipes.
* I would like to try some Indian recipes. Do you have any suggestions.
* I'm on a gluten free diet. Can you suggest some desserts.
* The chatbot has 'memory', so you can ask something like:
    * I don't like {name of meal it just suggested}, can you please provide another suggestion to replace it
* The system prompt constraints the chatbot to only respond to questions relevant to meal planning, so you can try unrelated questions like:
    * What is CodeMash 
        * It should reply "Sorry, I don't have the information requested."