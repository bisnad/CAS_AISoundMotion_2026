# 
repos=( "unloop-music" "unloop-percussion" "unloop-n64" "unloop-birds" "unloop-choir" "unloop-machines" "nesquik" "unloop-opera")
for repo in "${repos[@]}"
do
    echo "Updating $repo"
    git remote add --fetch $repo https://huggingface.co/spaces/hugggof/$repo
    git push --force $repo main
done

# https://huggingface.co/spaces/hugggof/unloop-music
# git push --space-percussion main 