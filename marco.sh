marco(){
    
    WXDIR="data/AvatarCap/lmx_ml"
    # cpgwx is an alias of scp command
    for file in "$@"; do
        cpgwx "$file" "$WXDIR"
    done
    
}

polo(){
    
    WXDIR="data/AvatarCap/lmx_ml"
    # cpgwx is an alias of scp command
    for file in "$@"; do
        cpgwo "$WXDIR/$file"
    done
    
}

marcoda(){
    
    DIR="YOLO"
    # cpgwx is an alias of scp command
    for file in "$@"; do
        # if file is a directory
        if [ -d "$file" ]; then
            cpdar "$file" "$DIR"
        else
            cpda "$file" "$DIR"
        fi
    done
    
}

poloda(){
    
    DIR="YOLO"
    if [ -d "${DIR}/$1" ]; then
        cpdor "${DIR}/$1" .
    else
        cpdo "${DIR}/$1" .
    fi
    
}
