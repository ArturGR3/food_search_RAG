from application.recipe_recommendation import get_recommendations
import gradio as gr

def format_recipe(recipe):
    return f"""
# {recipe['Title']}

## Ingredients:
{recipe['Ingredients'][:300]}...

## Instructions:
{recipe['Instructions'][:300]}...
    """

def recommend(query):
    recommendations = get_recommendations(query)
    
    outputs = []
    for recipe in recommendations:
        outputs.extend([
            recipe['Image_Path'],
            format_recipe(recipe)
        ])
    
    # Pad outputs if less than 3 recommendations
    while len(outputs) < 6:
        outputs.extend([None, "No recommendation"])
    
    return outputs

iface = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(lines=2, placeholder="Enter your recipe query here..."),
    outputs=[
        gr.Image(type="filepath", label="Recipe 1"),
        gr.Markdown(label="Details 1"),
        gr.Image(type="filepath", label="Recipe 2"),
        gr.Markdown(label="Details 2"),
        gr.Image(type="filepath", label="Recipe 3"),
        gr.Markdown(label="Details 3")
    ],
    title="Recipe Recommendation System",
    description="Enter a query to get recipe recommendations with images!",
    examples=[
        ["vegetarian pasta dish with tomatoes"],
        ["spicy chicken curry"],
        ["chocolate dessert for two"]
    ]
)

if __name__ == "__main__":
    iface.launch()