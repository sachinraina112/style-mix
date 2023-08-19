#/bin/bash
echo "Serving Inference"
{ # try

    python serve.py
    #save your output

} || { python3 serve.py
}

