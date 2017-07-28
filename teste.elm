import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (onClick , onInput)
import Array exposing (..)
import Random
import Dict

-- MAIN
main =  Html.beginnerProgram { model = model, view = view, update = update }

-- MODEL
type alias Model ={message:String, running:Bool , lastrow:Int, lastcol:Int, head:Dict.Dict String String, body:Dict.Dict String Float}

type Msg = Running | Addrow | Addcol | Changeh String String | Changeb String String

ir = 3
ic = 3

model = Model "ok" False ir ic (inithead ic) (initbody ir ic)



inithead ncols = Dict.fromList (Array.toList (initialize ncols (\c -> (toString (c + 1), "Atributo "++ toString (c + 1)))))

row r c = List.map (\n->(r,n)) (List.range 1 c)
col r c  = List.concatMap (\n->(row n c)) (List.range 1 r)
intinttostr l = List.map (\n-> (toString (Tuple.first n)) ++ "." ++ (toString (Tuple.second n))) l
strtodict l = Dict.fromList (List.map (\n->(n,0)) l)

initbody r c = col r c |> intinttostr |> strtodict



-- UPDATE
update : Msg -> Model -> Model
update msg model =
  case msg of
    Running -> {model | running = not model.running}
    Addrow -> {model
      | lastrow =  model.lastrow + 1
      , body = initbody (model.lastrow + 1) (model.lastcol)
      }
    Addcol -> {model
      | lastcol =  model.lastcol + 1
      , head = inithead (model.lastcol + 1)
      , body = initbody (model.lastrow) (model.lastcol + 1)
      }
    Changeh i s -> {model
      | message = i ++ s
      , head = Dict.insert i s model.head
      }
    Changeb i s -> {model
      | message = i ++ s
      , body = Dict.insert i (stf s) model.body
      }


-- VIEW
view : Model -> Html Msg
view m =  section [] [headerview m , matrixview m]

headerview m =
  div []
    [
    button [ onClick Addrow ] [ text "Add Row" ]
    , button [ onClick Addcol ] [ text "Add Column" ]
    , button [ onClick Running ] [ text "Run/Stop" ]
    , div [] [
              text (--toString(m.message)
              --++ toString(m.lastrow)
              --++ toString(m.lastcol)
              --++ toString(m.running) ++
               toString(m.body))
             ]
    ]

stf s =
  case String.toFloat s of
    Ok val -> val
    Err er -> 0


headtohtml l = List.map2 (\i h -> th [] [input [placeholder h, onInput (Changeh (i))][]]) (Dict.keys l) (Dict.values l)

toHTMLtable head body ids =
  case List.take (List.length (Dict.keys head)) body of
    [] -> []
    nb ->
      let
        ni = List.take (List.length (Dict.keys head)) ids
      in
        (tr[](List.map2 (\b i-> td [] [input [placeholder b, onInput (Changeb (i))][]]) nb ni))
        :: (toHTMLtable head (List.drop (List.length (Dict.keys head)) body) (List.drop (List.length (Dict.keys head)) ids))

matrixview m = table[] (List.concat ((headtohtml (m.head)) :: [toHTMLtable (m.head) (List.map (\n-> toString n) (Dict.values (m.body))) (Dict.keys (m.body))]))
