import gradio as gr
from recipe_recommendation import get_recommendations
from db import (setup_database, create_session, update_session_activity, store_user_query, 
                store_returned_recipes, store_user_click, save_feedback, show_recent_query_adjustments)
from query_preprocessor import preprocess_query, PreprocessedQuery
from llm_factory import LLMFactory

def recommend(query, session_id, llm_factory):
    # Preprocess the query
    preprocessed_query = preprocess_query(query, llm_factory, session_id)
    
    if not preprocessed_query.is_recipe_request:
        return [], "Sorry, your query doesn't seem to be a recipe request. Please try again with a recipe-related query."
    
    # Store user query and get query_id
    query_id = store_user_query(session_id, preprocessed_query.adjusted_query)
    
    # Update session activity
    update_session_activity(session_id)
    
    # Get recommendations using the adjusted query
    recommendations = get_recommendations(preprocessed_query.adjusted_query)
    
    # Filter out recipes containing excluded ingredients
    filtered_recommendations = [
        recipe for recipe in recommendations 
        if not any(ingredient.lower() in recipe['Ingredients'].lower() for ingredient in preprocessed_query.excluded_ingredients)
    ]
    
    # Limit to top 10 recommendations
    filtered_recommendations = filtered_recommendations[:10]
    
    # Store returned recipes
    store_returned_recipes(session_id, query_id, filtered_recommendations)
    
    # Prepare feedback about query adjustments
    adjustment_feedback = f"Adjusted query: {preprocessed_query.adjusted_query}\n\n"
    if preprocessed_query.adjustments:
        adjustment_feedback += "Adjustments made:\n" + "\n".join(f"- {adj}" for adj in preprocessed_query.adjustments)
    if preprocessed_query.excluded_ingredients:
        adjustment_feedback += f"\n\nExcluded ingredients:\n" + "\n".join(f"- {ing}" for ing in preprocessed_query.excluded_ingredients)
    
    return filtered_recommendations, adjustment_feedback

def create_recipe_component():
    with gr.Column():
        title = gr.Markdown()
        image = gr.Image(type="filepath", label="")
        with gr.Row():
            ingredients_btn = gr.Button("Ingredients", scale=1)
            instructions_btn = gr.Button("Instructions", scale=1)
        details = gr.Markdown(visible=True)
        with gr.Row():
            thumbs_up = gr.Button("👍", scale=1)
            thumbs_down = gr.Button("👎", scale=1)
        feedback_input = gr.Textbox(label="Feedback", visible=False, placeholder="Please provide more details about your feedback...")
    
    return title, image, ingredients_btn, instructions_btn, details, thumbs_up, thumbs_down, feedback_input

def show_details(recipes, index, detail_type, current_content, session_id):
    if index < len(recipes) and recipes[index]:
        recipe_title = recipes[index]['Title']
        new_content = recipes[index][detail_type]
        # Store user click
        store_user_click(session_id, recipe_title, f"View {detail_type}")
        # Toggle visibility: if current content is the same as new, clear it
        if current_content.strip() == new_content.strip():
            return ""
        else:
            return new_content
    return ""

def handle_feedback(username, recipe_title, session_id, is_positive):
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
    save_feedback(username, recipe_title, feedback)
    store_user_click(session_id, recipe_title, "Detailed Feedback")
    update_session_activity(session_id)
    return gr.update(visible=False), f"Thank you for your feedback, {username}!"

def display_adjustments():
    df = show_recent_query_adjustments()
    markdown_output = "# Recent Query Adjustments\n\n"
    for _, row in df.iterrows():
        markdown_output += f"**Timestamp:** {row['timestamp']}\n\n"
        markdown_output += f"**Original Query:** {row['original_query']}\n\n"
        markdown_output += f"**Adjusted Query:** {row['adjusted_query']}\n\n"
        markdown_output += f"**Excluded Ingredients:** {', '.join(row['excluded_ingredients'])}\n\n"
        markdown_output += "---\n\n"
    return markdown_output

def main():
    try:
        setup_database()
        global llm_factory
        llm_factory = LLMFactory("groq")  # Or whichever provider you prefer
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        return

    with gr.Blocks(title="Recipe Recommendation System") as iface:
        username = gr.State("")
        session_id = gr.State("")
        
        # Login section
        with gr.Column(visible=True) as login_section:
            gr.Markdown("# Welcome to the Recipe Recommendation System")
            username_input = gr.Textbox(label="Please enter your name:")
            login_button = gr.Button("Start")
        
        # Main application section (initially hidden)
        with gr.Column(visible=False) as main_section:
            welcome_msg = gr.Markdown("Welcome!")
            gr.Markdown("Enter a query to get recipe recommendations!")
            
            with gr.Row():
                query_input = gr.Textbox(lines=1, placeholder="Enter your recipe query here...", scale=3)
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
            
            # Recipe components (initially hidden)
            with gr.Column(visible=False) as recipe_section:
                with gr.Row():
                    recipe1_title, recipe1_img, r1_ing_btn, r1_ins_btn, r1_details, r1_up, r1_down, r1_feedback = create_recipe_component()
                    recipe2_title, recipe2_img, r2_ing_btn, r2_ins_btn, r2_details, r2_up, r2_down, r2_feedback = create_recipe_component()
                    recipe3_title, recipe3_img, r3_ing_btn, r3_ins_btn, r3_details, r3_up, r3_down, r3_feedback = create_recipe_component()
            
            feedback_text = gr.Markdown()
            
            recipes = gr.State([])

        def on_login(name):
            new_session_id = create_session(name)
            return (
                gr.update(visible=False),  # Hide login section
                gr.update(visible=True),   # Show main section
                f"# Welcome, {name}, to the Recipe Recommendation System!",
                name,  # Update username state
                new_session_id  # Update session_id state
            )
        
        login_button.click(
            on_login,
            inputs=[username_input],
            outputs=[login_section, main_section, welcome_msg, username, session_id]
        )
        
        def update_recipes_and_feedback(query, session_id):
            recipes, adjustment_feedback = recommend(query, session_id, llm_factory)
            
            # Ensure we always have 3 recipes, even if fewer are returned
            titles = [recipe.get('Title', '') for recipe in recipes[:3]]
            images = [recipe.get('Image_Path', None) for recipe in recipes[:3]]
            
            # Pad the lists if there are fewer than 3 recipes
            titles += [''] * (3 - len(titles))
            images += [None] * (3 - len(images))
            
            return [
                titles[0], titles[1], titles[2],  # 3 title outputs
                images[0], images[1], images[2],  # 3 image outputs
                recipes,  # State for all recipes
                gr.update(visible=True, value=adjustment_feedback),  # Adjustment feedback
                gr.update(visible=True)  # Make recipe section visible
            ]

        # Update the submit button click event
        submit_btn.click(
            update_recipes_and_feedback, 
            inputs=[query_input, session_id], 
            outputs=[
                recipe1_title, recipe2_title, recipe3_title,
                recipe1_img, recipe2_img, recipe3_img, 
                recipes, adjustment_feedback, recipe_section
            ]
        )
        
        # Add event for pressing Enter in the query input
        query_input.submit(
            update_recipes_and_feedback,
            inputs=[query_input, session_id],
            outputs=[
                recipe1_title, recipe2_title, recipe3_title,
                recipe1_img, recipe2_img, recipe3_img, 
                recipes, adjustment_feedback, recipe_section
            ]
        )
        
        for i, (ing_btn, ins_btn, details, up_btn, down_btn, feedback_input) in enumerate([
            (r1_ing_btn, r1_ins_btn, r1_details, r1_up, r1_down, r1_feedback),
            (r2_ing_btn, r2_ins_btn, r2_details, r2_up, r2_down, r2_feedback),
            (r3_ing_btn, r3_ins_btn, r3_details, r3_up, r3_down, r3_feedback)
        ]):
            ing_btn.click(
                lambda recipes, current, session_id, i=i: show_details(recipes, i, "Ingredients", current, session_id), 
                inputs=[recipes, details, session_id], 
                outputs=[details]
            )
            ins_btn.click(
                lambda recipes, current, session_id, i=i: show_details(recipes, i, "Instructions", current, session_id), 
                inputs=[recipes, details, session_id], 
                outputs=[details]
            )
            up_btn.click(
                lambda username, title, session_id, i=i: handle_feedback(username, title, session_id, True),
                inputs=[username, recipe1_title if i == 0 else (recipe2_title if i == 1 else recipe3_title), session_id],
                outputs=[up_btn, down_btn, feedback_input, feedback_text]
            )
            down_btn.click(
                lambda username, title, session_id, i=i: handle_feedback(username, title, session_id, False),
                inputs=[username, recipe1_title if i == 0 else (recipe2_title if i == 1 else recipe3_title), session_id],
                outputs=[up_btn, down_btn, feedback_input, feedback_text]
            )
            feedback_input.submit(
                lambda username, feedback, title, session_id, i=i: submit_feedback(username, title, feedback, session_id),
                inputs=[username, feedback_input, recipe1_title if i == 0 else (recipe2_title if i == 1 else recipe3_title), session_id],
                outputs=[feedback_input, feedback_text]
            )
    
    iface.launch()

if __name__ == "__main__":
    main()