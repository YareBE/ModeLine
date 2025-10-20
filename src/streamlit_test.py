import streamlit as st

st.title("EXAMPLE OF AN EASY STREAMLIT PAGE")
st.subheader("This code is only 20 lines long by the way")
st.divider()
reset = st.button("Reset", type="primary")
if reset:
    st.session_state.button1 = None
if not st.session_state.button1:
    st.session_state.button1 = st.button("Press me")
if st.session_state.button1:
    st.write("Hello!😊")
    st.session_state.user_score = st.slider("How much did you like the experience?", 0, 100, 40)
    if st.session_state.user_score < 50:
        st.write("Are you sure?😥")
    elif st.session_state.user_score < 80:
        st.write("Fair enough!")
    else: st.write("Perfect👌")
else:
    st.write("Goodbye!")