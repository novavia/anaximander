from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).parent

max_score = 100
test_name = "Python Challenge"
students = [
    {"name": "Sandrine",  "score": 100},
    {"name": "Gergeley", "score": 87},
    {"name": "Frieda", "score": 92},
    {"name": "Fritz", "score": 40},
    {"name": "Sirius", "score": 75},
]

environment = Environment(loader=FileSystemLoader("templates/"))
template = environment.get_template("message.txt")

for student in students:
    filename = f"message_{student['name'].lower()}.txt"
    fpath = ROOT / "messages" / filename
    content = template.render(
        student,
        max_score=max_score,
        test_name=test_name
    )
    with open(fpath, mode="w", encoding="utf-8") as message:
        message.write(content)
        print(f"... wrote {filename}")


results_filename = "students_results.html"
results_fpath = ROOT / results_filename
results_template = environment.get_template("results.html")
context = {
    "students": students,
    "test_name": test_name,
    "max_score": max_score,
}
with open(results_fpath, mode="w", encoding="utf-8") as results:
    results.write(results_template.render(context))
    print(f"... wrote {results_filename}")
