import gradio as gr
from recipe_recommendation import get_recommendations
from src.reporting.db import (
    setup_database, create_session, update_session_activity, store_user_query,
    store_retrieved_recipes, store_assessed_recipes, store_recommended_recipes,
    store_user_click, save_feedback, show_recent_query_adjustments, store_bad_image_feedback
)
from src.user_query_preprocessing.query_preprocessor import preprocess_query
from src.utils.llm_factory import LLMFactory
from gradio import Progress
import os

def generate_adjustment_feedback(preprocessed_query):
    feedback = f"Adjusted query: {preprocessed_query.adjusted_query}\n\n"
    if preprocessed_query.adjustments:
        feedback += "Adjustments made:\n" + "\n".join(f"- {adj}" for adj in preprocessed_query.adjustments)
    if preprocessed_query.excluded_ingredients:
        feedback += f"\n\nExcluded ingredients:\n" + "\n".join(f"- {ing}" for ing in preprocessed_query.excluded_ingredients)
    return feedback

def recommend(query, session_id, llm_factory, progress=gr.Progress()):

    # 1. Check if the query is a valid recipe request and preprocess it
    preprocessed_query = preprocess_query(query, llm_factory, session_id)
    
    if not preprocessed_query.is_recipe_request:
        return [], "Sorry, your query doesn't seem to be a recipe request. Please try again with a recipe-related query."
    
    # 2. Store the query and update session activity
    query_id = store_user_query(session_id, preprocessed_query.adjusted_query)
    update_session_activity(session_id)
    
    # 3. Generate adjustment feedback of user query preprocessing
    adjustment_feedback = generate_adjustment_feedback(preprocessed_query)
    yield [], adjustment_feedback, gr.update(visible=False)
    
    progress(0.3, desc="Retrieving recipes...")
    recommendations = get_recommendations(preprocessed_query.adjusted_query, preprocessed_query.excluded_ingredients, llm_factory, max_recommendations=3, min_relevance_score=5)
    
    progress(0.6, desc="Processing recommendations...")
    store_recommendations(query_id, recommendations)
    
    progress(1.0, desc="Finalizing results...")
    recommended_recipes = recommendations.get("recommended_recipes", [])
    
    if not recommended_recipes:
        yield [], "No matching recipes found. Please try a different query.", gr.update(visible=False)
    else:
        yield recommended_recipes, adjustment_feedback, gr.update(visible=True)


def store_recommendations(query_id, recommendations):
    retrieved_recipes = recommendations.get("retrieved_recipes", [])
    assessed_recipes = recommendations.get("assessed_recipes", [])
    recommended_recipes = recommendations.get("recommended_recipes", [])
    
    store_retrieved_recipes(query_id, retrieved_recipes)
    store_assessed_recipes(query_id, assessed_recipes)
    store_recommended_recipes(query_id, recommended_recipes)

def create_recipe_component():
    with gr.Column():
        title = gr.Markdown()
        image = gr.Image(type="filepath", label="")
        with gr.Row():
            ingredients_btn = gr.Button("Ingredients", scale=1)
            instructions_btn = gr.Button("Instructions", scale=1)
            bad_image_btn = gr.Button("Bad Image", scale=1)
        details = gr.Markdown(visible=True)
        with gr.Row():
            thumbs_up = gr.Button("👍", scale=1)
            thumbs_down = gr.Button("👎", scale=1)
        feedback_input = gr.Textbox(label="Feedback", visible=False, placeholder="Please provide more details about your feedback...")
    
    return title, image, ingredients_btn, instructions_btn, bad_image_btn, details, thumbs_up, thumbs_down, feedback_input

def handle_bad_image(username, recipe_title, session_id, current_state):
    """
    Toggle the bad image report state and update the database
    """
    if current_state == "Bad Image":
        store_bad_image_feedback(username, recipe_title)
        store_user_click(session_id, recipe_title, "Bad Image")
        update_session_activity(session_id)
        new_state = "Bad Image Reported"
        message = f"Thank you for reporting the bad image for {recipe_title}, {username}. We'll look into it."
    else:
        new_state = "Bad Image"
        message = f"Report for {recipe_title} has been cancelled."

    return gr.update(value=new_state), message

def show_details(recipes, index, detail_type, current_content, session_id):
    """
    Show details of the recipe when the user clicks on the buttons
    """
    if index < len(recipes) and recipes[index]:
        recipe_title = recipes[index]['Title']
        
        # Map the detail_type to the correct key in the recipe dictionary
        detail_key = {
            "Ingredients": "Cleaned_Ingredients",
            "Instructions": "Instructions"
        }.get(detail_type, detail_type)
        
        new_content = recipes[index].get(detail_key, "Details not available")
        store_user_click(session_id, recipe_title, f"View {detail_type}")
        return "" if current_content.strip() == new_content.strip() else new_content
    return ""

def handle_feedback(username, recipe_title, session_id, is_positive):
    """
    Handle the feedback of the user when they click on the thumbs up or thumbs down button
    """
    feedback_type = "Liked" if is_positive else "Disliked"
    save_feedback(username, recipe_title, feedback_type)
    store_user_click(session_id, recipe_title, feedback_type)
    update_session_activity(session_id)
    
    if is_positive:
        return (
            gr.update(value="👍 Liked!", variant="secondary"),
            gr.update(value="👎"),
            gr.update(visible=False),
            f"Thank you for your positive feedback, {username}!"
        )
    else:
        return (
            gr.update(value="👍"),
            gr.update(value="👎 Disliked!", variant="secondary"),
            gr.update(visible=True),
            f"We're sorry to hear that, {username}. Please provide more details below:"
        )

def submit_feedback(username, recipe_title, feedback, session_id):
    """
    Save the feedback of the user when they click on the thumbs up or thumbs down button
    """
    save_feedback(username, recipe_title, feedback)
    store_user_click(session_id, recipe_title, "Detailed Feedback")
    update_session_activity(session_id)
    return gr.update(visible=False), f"Thank you for your feedback, {username}!"

def display_adjustments():
    """
    Display the recent query adjustments
    """
    df = show_recent_query_adjustments()
    markdown_output = "# **Recent Query Adjustments**\n\n"
    for _, row in df.iterrows():
        markdown_output += f"**Timestamp:** {row['timestamp']}\n\n"
        markdown_output += f"**Original Query:** **{row['original_query']}**\n\n"
        markdown_output += f"**Adjusted Query:** **{row['adjusted_query']}**\n\n"
        markdown_output += f"**Excluded Ingredients:** {', '.join(row['excluded_ingredients'])}\n\n"
        markdown_output += "---\n\n"
    return markdown_output


def generate_recipe_output(recipes):
    """
    Generate recipe output if the user clicks on the submit button and 
    the query is a valid recipe request
    """
    output = [
        recipes[0]['Title'] if len(recipes) > 0 else "",
        recipes[1]['Title'] if len(recipes) > 1 else "",
        recipes[2]['Title'] if len(recipes) > 2 else "",
        recipes[0]['Image_Path'] if len(recipes) > 0 else None,
        recipes[1]['Image_Path'] if len(recipes) > 1 else None,
        recipes[2]['Image_Path'] if len(recipes) > 2 else None,
    ]
    
    # Debug: Print image paths
    for i, path in enumerate(output[3:]):
        if path:
            print(f"Image path for recipe {i+1}: {path}")
            if not os.path.exists(path):
                print(f"Warning: Image file does not exist: {path}")
    
    return output

def update_recipes_and_feedback(query, session_id, progress=gr.Progress(), llm_model='openai'):
    """
    Update the recipes and feedback when the user clicks on the submit button
    """
    llm_factory = LLMFactory(llm_model)
    generator = recommend(query, session_id, llm_factory, progress)
    
    recipes, adjustment_feedback, recipe_section_update = next(generator)
    yield generate_empty_recipe_output() + [
        recipes,
        gr.update(visible=True, value=adjustment_feedback),
        recipe_section_update,
        gr.update(value="Processing...", interactive=False)  # Keep processing
    ]
    
    try:
        recipes, _, recipe_section_update = next(generator)
        if recipes:
            yield generate_recipe_output(recipes) + [
                recipes,
                gr.update(visible=True, value=adjustment_feedback),
                recipe_section_update,
                gr.update(value="Get Recommendations", interactive=True)  # Reset button
            ]
        else:
            yield generate_empty_recipe_output() + [
                [],
                gr.update(visible=True, value=adjustment_feedback),
                recipe_section_update,
                gr.update(value="Get Recommendations", interactive=True)  # Reset button
            ]
    except StopIteration:
        yield generate_empty_recipe_output() + [
            [],
            gr.update(visible=True, value="An error occurred while processing your request."),
            gr.update(visible=False),
            gr.update(value="Get Recommendations", interactive=True)  # Reset button
        ]

def generate_empty_recipe_output():
    """
    Generate empty recipe output if the user clicks on the submit button and 
    the query is not a valid recipe request
    """
    return ["", "", "", None, None, None]
    
def create_login_section():
    """
    Create the login section of the recipe recommendation system
    Where the user can enter their name to start the session
    """
    with gr.Column(visible=True) as login_section:
        gr.Markdown("# Welcome to the Recipe Recommendation System")
        username_input = gr.Textbox(label="Please enter your name:", scale=1)
        login_button = gr.Button("Start", scale=1)
    return login_section, username_input, login_button

def create_main_section():
    """
    Create the main section of the recipe recommendation system
    Where the user can enter their query to get recipe recommendations
    """
    with gr.Column(visible=False) as main_section:
        welcome_msg = gr.Markdown("Welcome!")
        gr.Markdown("Enter a query to get recipe recommendations!")
        
        with gr.Row():
            query_input = gr.Textbox(lines=1, placeholder="Enter your recipe query here...", scale=2)
            submit_btn = gr.Button("Get Recommendations", variant="primary", scale=1)
        
        example_queries = gr.Examples(
            examples=[
                "vegetarian pasta dish with tomatoes",
                "spicy chicken curry without nuts",
                "chocolate dessert for two, no dairy",
                "gluten-free pizza recipe",
                "salad with no chicken"
            ],
            inputs=[query_input],
            label="Example Queries"
        )
        
        adjustment_feedback = gr.Markdown(visible=False)
        
    with gr.Column(visible=False) as recipe_section:
        with gr.Row():
            recipe1_components = create_recipe_component()
            recipe2_components = create_recipe_component()
            recipe3_components = create_recipe_component()
        
        feedback_text = gr.Markdown()
        recipes = gr.State([])
    
    components = {
        'welcome_msg': welcome_msg,
        'query_input': query_input,
        'submit_btn': submit_btn,
        'adjustment_feedback': adjustment_feedback,
        'recipe1': recipe1_components,
        'recipe2': recipe2_components,
        'recipe3': recipe3_components,
        'feedback_text': feedback_text,
        'recipes': recipes
    }
    
    return main_section, recipe_section, components


def on_login(name):
    """
    Create a new session for the user when they click on the login button
    """
    new_session_id = create_session(name)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"# Welcome, {name}, to the Recipe Recommendation System!",
        name,
        new_session_id
    )

def on_button_click(btn):
    """
    Update the button to show that the system is processing the query
    """
    return gr.update(value="Processing...", interactive=False)

def setup_recipe_component_handlers(recipe_components, index, components, username, session_id):
    """
    Setup the handlers for the recipe components that show the details of the recipe
    """
    ing_btn, ins_btn, bad_img_btn, details, up_btn, down_btn, feedback_input = recipe_components[2:]
    recipe_title = recipe_components[0]
    
    ing_btn.click(
        lambda recipes, current, session: show_details(recipes, index, "Ingredients", current, session),
        inputs=[components['recipes'], details, session_id],
        outputs=[details]
    )
    ins_btn.click(
        lambda recipes, current, session: show_details(recipes, index, "Instructions", current, session),
        inputs=[components['recipes'], details, session_id],
        outputs=[details]
    )
    up_btn.click(
        lambda user, title, session: handle_feedback(user, title, session, True),
        inputs=[username, recipe_title, session_id],
        outputs=[up_btn, down_btn, feedback_input, components['feedback_text']]
    )
    down_btn.click(
        lambda user, title, session: handle_feedback(user, title, session, False),
        inputs=[username, recipe_title, session_id],
        outputs=[up_btn, down_btn, feedback_input, components['feedback_text']]
    )
    feedback_input.submit(
        lambda user, feedback, title, session: submit_feedback(user, title, feedback, session),
        inputs=[username, feedback_input, recipe_title, session_id],
        outputs=[feedback_input, components['feedback_text']]
    )
    bad_img_btn.click(
        handle_bad_image,
        inputs=[username, recipe_title, session_id, bad_img_btn],
        outputs=[bad_img_btn, components['feedback_text']]
    )


def setup_event_handlers(login_section, main_section, recipe_section, components, username, session_id):
    login_section[2].click(
        on_login,
        inputs=[login_section[1]],
        outputs=[login_section[0], main_section, components['welcome_msg'], username, session_id],
        show_progress="minimal"
    )
    
    components['submit_btn'].click(
        on_button_click,
        inputs=[components['submit_btn']],
        outputs=[components['submit_btn']],
        queue=False
    ).then(
        update_recipes_and_feedback,
        inputs=[components['query_input'], session_id],
        outputs=[
            components['recipe1'][0], components['recipe2'][0], components['recipe3'][0],
            components['recipe1'][1], components['recipe2'][1], components['recipe3'][1],
            components['recipes'], components['adjustment_feedback'], recipe_section,
            components['submit_btn']
        ],
        show_progress=True
    )
    
    components['query_input'].submit(
        on_button_click,
        inputs=[components['submit_btn']],
        outputs=[components['submit_btn']],
        queue=False
    ).then(
        update_recipes_and_feedback,
        inputs=[components['query_input'], session_id],
        outputs=[
            components['recipe1'][0], components['recipe2'][0], components['recipe3'][0],
            components['recipe1'][1], components['recipe2'][1], components['recipe3'][1],
            components['recipes'], components['adjustment_feedback'], recipe_section,
            components['submit_btn']
        ],
        show_progress=True
    )
    
    setup_recipe_component_handlers(components['recipe1'], 0, components, username, session_id)
    setup_recipe_component_handlers(components['recipe2'], 1, components, username, session_id)
    setup_recipe_component_handlers(components['recipe3'], 2, components, username, session_id)


def main():
    """
    Main function to run the recipe recommendation system
    """
    llm_model = "openai"
    try:
        setup_database()
        global llm_factory
        llm_factory = LLMFactory(llm_model)
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        return

    with gr.Blocks(title="Recipe Recommendation System") as iface:
        username = gr.State("")
        session_id = gr.State("")
        
        login_section = create_login_section()
        main_section, recipe_section, components = create_main_section()
        
        setup_event_handlers(login_section, main_section, recipe_section, components, username, session_id)
    
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
    
# test get_recommendations function
import pandas as pd 
df = pd.read_csv("data/recipes.csv")
recommendations = get_recommendations("chicken pasta", df, llm_factory, max_recommendations=3, min_relevance_score=5)