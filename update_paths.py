import os

md_dir = '/home/youssef/المستندات/مشاريع لبيةالخيرة/ترجمة موقع معاجلة اللغة الطبيعية/NLP-Course'
old_base = '/NLP-Course'
new_base = '/NLP-Course-Ar'

def replace_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We want to replace /NLP-Course/ with /NLP-Course-Ar/
    # And also just "/NLP-Course" (without trailing slash) in baseurl if needed, but we can do that separately or carefully.
    new_content = content.replace('/NLP-Course/', '/NLP-Course-Ar/')
    new_content = new_content.replace('"/NLP-Course"', '"/NLP-Course-Ar"')
    new_content = new_content.replace('https://ingeotec.github.io/NLP-Course/', 'https://YoussefAI26.github.io/NLP-Course-Ar/')
    new_content = new_content.replace('https://ingeotec.github.io/NLP-Course', 'https://YoussefAI26.github.io/NLP-Course-Ar')
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated paths in {os.path.basename(filepath)}")

# Update in topics/*.md
topics_dir = os.path.join(md_dir, 'topics')
for filename in os.listdir(topics_dir):
    if filename.endswith('.md'):
        replace_in_file(os.path.join(topics_dir, filename))

# Update in other files
replace_in_file(os.path.join(md_dir, '_config.yml'))
replace_in_file(os.path.join(md_dir, 'README.md'))
replace_in_file(os.path.join(md_dir, 'index.md'))
replace_in_file(os.path.join(md_dir, '_includes/nav_footer_custom.html'))
replace_in_file(os.path.join(md_dir, '_includes/head_custom.html'))
